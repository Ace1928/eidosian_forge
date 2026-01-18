import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
class Watcher(object):
    ignored_dirs = {u'.git', u'__pycache__', u'.idea', u'node_modules', u'.metadata'}
    accepted_file_extensions = ()
    target_time_for_single_scan = 2.0
    target_time_for_notification = 4.0
    print_poll_time = False
    max_recursion_level = 10

    def __init__(self, accept_directory=None, accept_file=None):
        """
        :param Callable[str, bool] accept_directory:
            Callable that returns whether a directory should be watched.
            Note: if passed it'll override the `ignored_dirs`

        :param Callable[str, bool] accept_file:
            Callable that returns whether a file should be watched.
            Note: if passed it'll override the `accepted_file_extensions`.
        """
        self._path_watchers = set()
        self._disposed = threading.Event()
        if accept_directory is None:
            accept_directory = lambda dir_path: basename(dir_path) not in self.ignored_dirs
        if accept_file is None:
            accept_file = lambda path_name: not self.accepted_file_extensions or path_name.endswith(self.accepted_file_extensions)
        self.accept_file = accept_file
        self.accept_directory = accept_directory
        self._single_visit_info = _SingleVisitInfo()

    @property
    def accept_directory(self):
        return self._accept_directory

    @accept_directory.setter
    def accept_directory(self, accept_directory):
        self._accept_directory = accept_directory
        for path_watcher in self._path_watchers:
            path_watcher.accept_directory = accept_directory

    @property
    def accept_file(self):
        return self._accept_file

    @accept_file.setter
    def accept_file(self, accept_file):
        self._accept_file = accept_file
        for path_watcher in self._path_watchers:
            path_watcher.accept_file = accept_file

    def dispose(self):
        self._disposed.set()

    @property
    def path_watchers(self):
        return tuple(self._path_watchers)

    def set_tracked_paths(self, paths):
        """
        Note: always resets all path trackers to track the passed paths.
        """
        if not isinstance(paths, (list, tuple, set)):
            paths = (paths,)
        paths = sorted(set(paths), key=lambda path: -len(path))
        path_watchers = set()
        self._single_visit_info = _SingleVisitInfo()
        initial_time = time.time()
        for path in paths:
            sleep_time = 0.0
            path_watcher = _PathWatcher(path, self.accept_directory, self.accept_file, self._single_visit_info, max_recursion_level=self.max_recursion_level, sleep_time=sleep_time)
            path_watchers.add(path_watcher)
        actual_time = time.time() - initial_time
        pydev_log.debug('Tracking the following paths for changes: %s', paths)
        pydev_log.debug('Time to track: %.2fs', actual_time)
        pydev_log.debug('Folders found: %s', len(self._single_visit_info.visited_dirs))
        pydev_log.debug('Files found: %s', len(self._single_visit_info.file_to_mtime))
        self._path_watchers = path_watchers

    def iter_changes(self):
        """
        Continuously provides changes (until dispose() is called).

        Changes provided are tuples with the Change enum and filesystem path.

        :rtype: Iterable[Tuple[Change, str]]
        """
        while not self._disposed.is_set():
            initial_time = time.time()
            old_visit_info = self._single_visit_info
            old_file_to_mtime = old_visit_info.file_to_mtime
            changes = []
            append_change = changes.append
            self._single_visit_info = single_visit_info = _SingleVisitInfo()
            for path_watcher in self._path_watchers:
                path_watcher._check(single_visit_info, append_change, old_file_to_mtime)
            for entry in old_file_to_mtime:
                append_change((Change.deleted, entry))
            for change in changes:
                yield change
            actual_time = time.time() - initial_time
            if self.print_poll_time:
                print('--- Total poll time: %.3fs' % actual_time)
            if actual_time > 0:
                if self.target_time_for_single_scan <= 0.0:
                    for path_watcher in self._path_watchers:
                        path_watcher.sleep_time = 0.0
                else:
                    perc = self.target_time_for_single_scan / actual_time
                    if perc > 2.0:
                        perc = 2.0
                    elif perc < 0.5:
                        perc = 0.5
                    for path_watcher in self._path_watchers:
                        if path_watcher.sleep_time <= 0.0:
                            path_watcher.sleep_time = 0.001
                        new_sleep_time = path_watcher.sleep_time * perc
                        diff_sleep_time = new_sleep_time - path_watcher.sleep_time
                        path_watcher.sleep_time += diff_sleep_time / (3.0 * len(self._path_watchers))
                        if actual_time > 0:
                            self._disposed.wait(actual_time)
                        if path_watcher.sleep_time < 0.001:
                            path_watcher.sleep_time = 0.001
            diff = self.target_time_for_notification - actual_time
            if diff > 0.0:
                self._disposed.wait(diff)