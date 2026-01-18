import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
class INotify(FileDescriptor):
    """
    The INotify file descriptor, it basically does everything related
    to INotify, from reading to notifying watch points.

    @ivar _buffer: a L{bytes} containing the data read from the inotify fd.

    @ivar _watchpoints: a L{dict} that maps from inotify watch ids to
        watchpoints objects

    @ivar _watchpaths: a L{dict} that maps from watched paths to the
        inotify watch ids
    """
    _inotify = _inotify

    def __init__(self, reactor=None):
        FileDescriptor.__init__(self, reactor=reactor)
        self._fd = self._inotify.init()
        fdesc.setNonBlocking(self._fd)
        fdesc._setCloseOnExec(self._fd)
        self.connected = 1
        self._writeDisconnected = True
        self._buffer = b''
        self._watchpoints = {}
        self._watchpaths = {}

    def _addWatch(self, path, mask, autoAdd, callbacks):
        """
        Private helper that abstracts the use of ctypes.

        Calls the internal inotify API and checks for any errors after the
        call. If there's an error L{INotify._addWatch} can raise an
        INotifyError. If there's no error it proceeds creating a watchpoint and
        adding a watchpath for inverse lookup of the file descriptor from the
        path.
        """
        path = path.asBytesMode()
        wd = self._inotify.add(self._fd, path, mask)
        iwp = _Watch(path, mask, autoAdd, callbacks)
        self._watchpoints[wd] = iwp
        self._watchpaths[path] = wd
        return wd

    def _rmWatch(self, wd):
        """
        Private helper that abstracts the use of ctypes.

        Calls the internal inotify API to remove an fd from inotify then
        removes the corresponding watchpoint from the internal mapping together
        with the file descriptor from the watchpath.
        """
        self._inotify.remove(self._fd, wd)
        iwp = self._watchpoints.pop(wd)
        self._watchpaths.pop(iwp.path)

    def connectionLost(self, reason):
        """
        Release the inotify file descriptor and do the necessary cleanup
        """
        FileDescriptor.connectionLost(self, reason)
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError as e:
                log.err(e, "Couldn't close INotify file descriptor.")

    def fileno(self):
        """
        Get the underlying file descriptor from this inotify observer.
        Required by L{abstract.FileDescriptor} subclasses.
        """
        return self._fd

    def doRead(self):
        """
        Read some data from the observed file descriptors
        """
        fdesc.readFromFD(self._fd, self._doRead)

    def _doRead(self, in_):
        """
        Work on the data just read from the file descriptor.
        """
        self._buffer += in_
        while len(self._buffer) >= 16:
            wd, mask, cookie, size = struct.unpack('=LLLL', self._buffer[0:16])
            if size:
                name = self._buffer[16:16 + size].rstrip(b'\x00')
            else:
                name = None
            self._buffer = self._buffer[16 + size:]
            try:
                iwp = self._watchpoints[wd]
            except KeyError:
                continue
            path = iwp.path.asBytesMode()
            if name:
                path = path.child(name)
            iwp._notify(path, mask)
            if iwp.autoAdd and mask & IN_ISDIR and mask & IN_CREATE:
                new_wd = self.watch(path, mask=iwp.mask, autoAdd=True, callbacks=iwp.callbacks)
                self.reactor.callLater(0, self._addChildren, self._watchpoints[new_wd])
            if mask & IN_DELETE_SELF:
                self._rmWatch(wd)
                self.loseConnection()

    def _addChildren(self, iwp):
        """
        This is a very private method, please don't even think about using it.

        Note that this is a fricking hack... it's because we cannot be fast
        enough in adding a watch to a directory and so we basically end up
        getting here too late if some operations have already been going on in
        the subdir, we basically need to catchup.  This eventually ends up
        meaning that we generate double events, your app must be resistant.
        """
        try:
            listdir = iwp.path.children()
        except OSError:
            return
        for f in listdir:
            if f.isdir():
                wd = self.watch(f, mask=iwp.mask, autoAdd=True, callbacks=iwp.callbacks)
                iwp._notify(f, IN_ISDIR | IN_CREATE)
                self.reactor.callLater(0, self._addChildren, self._watchpoints[wd])
            if f.isfile():
                iwp._notify(f, IN_CREATE | IN_CLOSE_WRITE)

    def watch(self, path, mask=IN_WATCH_MASK, autoAdd=False, callbacks=None, recursive=False):
        """
        Watch the 'mask' events in given path. Can raise C{INotifyError} when
        there's a problem while adding a directory.

        @param path: The path needing monitoring
        @type path: L{FilePath}

        @param mask: The events that should be watched
        @type mask: L{int}

        @param autoAdd: if True automatically add newly created
                        subdirectories
        @type autoAdd: L{bool}

        @param callbacks: A list of callbacks that should be called
                          when an event happens in the given path.
                          The callback should accept 3 arguments:
                          (ignored, filepath, mask)
        @type callbacks: L{list} of callables

        @param recursive: Also add all the subdirectories in this path
        @type recursive: L{bool}
        """
        if recursive:
            for child in path.walk():
                if child.isdir():
                    self.watch(child, mask, autoAdd, callbacks, recursive=False)
        else:
            wd = self._isWatched(path)
            if wd:
                return wd
            mask = mask | IN_DELETE_SELF
            return self._addWatch(path, mask, autoAdd, callbacks)

    def ignore(self, path):
        """
        Remove the watch point monitoring the given path

        @param path: The path that should be ignored
        @type path: L{FilePath}
        """
        path = path.asBytesMode()
        wd = self._isWatched(path)
        if wd is None:
            raise KeyError(f'{path!r} is not watched')
        else:
            self._rmWatch(wd)

    def _isWatched(self, path):
        """
        Helper function that checks if the path is already monitored
        and returns its watchdescriptor if so or None otherwise.

        @param path: The path that should be checked
        @type path: L{FilePath}
        """
        path = path.asBytesMode()
        return self._watchpaths.get(path, None)