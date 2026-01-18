from __future__ import with_statement
from wandb_watchdog.utils import platform
import threading
import errno
import sys
import stat
import os
from wandb_watchdog.observers.api import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.events import (
class KqueueEmitter(EventEmitter):
    """
    kqueue(2)-based event emitter.

    .. ADMONITION:: About ``kqueue(2)`` behavior and this implementation

              ``kqueue(2)`` monitors file system events only for
              open descriptors, which means, this emitter does a lot of
              book-keeping behind the scenes to keep track of open
              descriptors for every entry in the monitored directory tree.

              This also means the number of maximum open file descriptors
              on your system must be increased **manually**.
              Usually, issuing a call to ``ulimit`` should suffice::

                  ulimit -n 1024

              Ensure that you pick a number that is larger than the
              number of files you expect to be monitored.

              ``kqueue(2)`` does not provide enough information about the
              following things:

              * The destination path of a file or directory that is renamed.
              * Creation of a file or directory within a directory; in this
                case, ``kqueue(2)`` only indicates a modified event on the
                parent directory.

              Therefore, this emitter takes a snapshot of the directory
              tree when ``kqueue(2)`` detects a change on the file system
              to be able to determine the above information.

    :param event_queue:
        The event queue to fill with events.
    :param watch:
        A watch object representing the directory to monitor.
    :type watch:
        :class:`watchdog.observers.api.ObservedWatch`
    :param timeout:
        Read events blocking timeout (in seconds).
    :type timeout:
        ``float``
    """

    def __init__(self, event_queue, watch, timeout=DEFAULT_EMITTER_TIMEOUT):
        EventEmitter.__init__(self, event_queue, watch, timeout)
        self._kq = select.kqueue()
        self._lock = threading.RLock()
        self._descriptors = KeventDescriptorSet()

        def walker_callback(path, stat_info, self=self):
            self._register_kevent(path, stat.S_ISDIR(stat_info.st_mode))
        self._snapshot = DirectorySnapshot(watch.path, watch.is_recursive, walker_callback)

    def _register_kevent(self, path, is_directory):
        """
        Registers a kevent descriptor for the given path.

        :param path:
            Path for which a kevent descriptor will be created.
        :param is_directory:
            ``True`` if the path refers to a directory; ``False`` otherwise.
        :type is_directory:
            ``bool``
        """
        try:
            self._descriptors.add(path, is_directory)
        except OSError as e:
            if e.errno == errno.ENOENT:
                pass
            else:
                raise

    def _unregister_kevent(self, path):
        """
        Convenience function to close the kevent descriptor for a
        specified kqueue-monitored path.

        :param path:
            Path for which the kevent descriptor will be closed.
        """
        self._descriptors.remove(path)

    def queue_event(self, event):
        """
        Handles queueing a single event object.

        :param event:
            An instance of :class:`watchdog.events.FileSystemEvent`
            or a subclass.
        """
        EventEmitter.queue_event(self, event)
        if event.event_type == EVENT_TYPE_CREATED:
            self._register_kevent(event.src_path, event.is_directory)
        elif event.event_type == EVENT_TYPE_MOVED:
            self._unregister_kevent(event.src_path)
            self._register_kevent(event.dest_path, event.is_directory)
        elif event.event_type == EVENT_TYPE_DELETED:
            self._unregister_kevent(event.src_path)

    def _queue_dirs_modified(self, dirs_modified, ref_snapshot, new_snapshot):
        """
        Queues events for directory modifications by scanning the directory
        for changes.

        A scan is a comparison between two snapshots of the same directory
        taken at two different times. This also determines whether files
        or directories were created, which updated the modified timestamp
        for the directory.
        """
        if dirs_modified:
            for dir_modified in dirs_modified:
                self.queue_event(DirModifiedEvent(dir_modified))
            diff_events = new_snapshot - ref_snapshot
            for file_created in diff_events.files_created:
                self.queue_event(FileCreatedEvent(file_created))
            for directory_created in diff_events.dirs_created:
                self.queue_event(DirCreatedEvent(directory_created))

    def _queue_events_except_renames_and_dir_modifications(self, event_list):
        """
        Queues events from the kevent list returned from the call to
        :meth:`select.kqueue.control`.

        .. NOTE:: Queues only the deletions, file modifications,
                  attribute modifications. The other events, namely,
                  file creation, directory modification, file rename,
                  directory rename, directory creation, etc. are
                  determined by comparing directory snapshots.
        """
        files_renamed = set()
        dirs_renamed = set()
        dirs_modified = set()
        for kev in event_list:
            descriptor = self._descriptors.get_for_fd(kev.ident)
            src_path = descriptor.path
            if is_deleted(kev):
                if descriptor.is_directory:
                    self.queue_event(DirDeletedEvent(src_path))
                else:
                    self.queue_event(FileDeletedEvent(src_path))
            elif is_attrib_modified(kev):
                if descriptor.is_directory:
                    self.queue_event(DirModifiedEvent(src_path))
                else:
                    self.queue_event(FileModifiedEvent(src_path))
            elif is_modified(kev):
                if descriptor.is_directory:
                    dirs_modified.add(src_path)
                else:
                    self.queue_event(FileModifiedEvent(src_path))
            elif is_renamed(kev):
                if descriptor.is_directory:
                    dirs_renamed.add(src_path)
                else:
                    files_renamed.add(src_path)
        return (files_renamed, dirs_renamed, dirs_modified)

    def _queue_renamed(self, src_path, is_directory, ref_snapshot, new_snapshot):
        """
        Compares information from two directory snapshots (one taken before
        the rename operation and another taken right after) to determine the
        destination path of the file system object renamed, and adds
        appropriate events to the event queue.
        """
        try:
            ref_stat_info = ref_snapshot.stat_info(src_path)
        except KeyError:
            if is_directory:
                self.queue_event(DirCreatedEvent(src_path))
                self.queue_event(DirDeletedEvent(src_path))
            else:
                self.queue_event(FileCreatedEvent(src_path))
                self.queue_event(FileDeletedEvent(src_path))
            return
        try:
            dest_path = absolute_path(new_snapshot.path_for_inode(ref_stat_info.st_ino))
            if is_directory:
                event = DirMovedEvent(src_path, dest_path)
                if self.watch.is_recursive:
                    for sub_event in event.sub_moved_events():
                        self.queue_event(sub_event)
                self.queue_event(event)
            else:
                self.queue_event(FileMovedEvent(src_path, dest_path))
        except KeyError:
            if is_directory:
                self.queue_event(DirDeletedEvent(src_path))
            else:
                self.queue_event(FileDeletedEvent(src_path))

    def _read_events(self, timeout=None):
        """
        Reads events from a call to the blocking
        :meth:`select.kqueue.control()` method.

        :param timeout:
            Blocking timeout for reading events.
        :type timeout:
            ``float`` (seconds)
        """
        return self._kq.control(self._descriptors.kevents, MAX_EVENTS, timeout)

    def queue_events(self, timeout):
        """
        Queues events by reading them from a call to the blocking
        :meth:`select.kqueue.control()` method.

        :param timeout:
            Blocking timeout for reading events.
        :type timeout:
            ``float`` (seconds)
        """
        with self._lock:
            try:
                event_list = self._read_events(timeout)
                files_renamed, dirs_renamed, dirs_modified = self._queue_events_except_renames_and_dir_modifications(event_list)
                new_snapshot = DirectorySnapshot(self.watch.path, self.watch.is_recursive)
                ref_snapshot = self._snapshot
                self._snapshot = new_snapshot
                if files_renamed or dirs_renamed or dirs_modified:
                    for src_path in files_renamed:
                        self._queue_renamed(src_path, False, ref_snapshot, new_snapshot)
                    for src_path in dirs_renamed:
                        self._queue_renamed(src_path, True, ref_snapshot, new_snapshot)
                    self._queue_dirs_modified(dirs_modified, ref_snapshot, new_snapshot)
            except OSError as e:
                if e.errno == errno.EBADF:
                    pass
                else:
                    raise

    def on_thread_stop(self):
        with self._lock:
            self._descriptors.clear()
            self._kq.close()