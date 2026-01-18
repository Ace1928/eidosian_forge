import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class WatchManager:
    """
    Provide operations for watching files and directories. Its internal
    dictionary is used to reference watched items. When used inside
    threaded code, one must instanciate as many WatchManager instances as
    there are ThreadedNotifier instances.

    """

    def __init__(self, exclude_filter=lambda path: False):
        """
        Initialization: init inotify, init watch manager dictionary.
        Raise OSError if initialization fails, raise InotifyBindingNotFoundError
        if no inotify binding was found (through ctypes or from direct access to
        syscalls).

        @param exclude_filter: boolean function, returns True if current
                               path must be excluded from being watched.
                               Convenient for providing a common exclusion
                               filter for every call to add_watch.
        @type exclude_filter: callable object
        """
        self._ignore_events = False
        self._exclude_filter = exclude_filter
        self._wmd = {}
        self._inotify_wrapper = INotifyWrapper.create()
        if self._inotify_wrapper is None:
            raise InotifyBindingNotFoundError()
        self._fd = self._inotify_wrapper.inotify_init()
        if self._fd < 0:
            err = 'Cannot initialize new instance of inotify, %s'
            raise OSError(err % self._inotify_wrapper.str_errno())

    def close(self):
        """
        Close inotify's file descriptor, this action will also automatically
        remove (i.e. stop watching) all its associated watch descriptors.
        After a call to this method the WatchManager's instance become useless
        and cannot be reused, a new instance must then be instanciated. It
        makes sense to call this method in few situations for instance if
        several independant WatchManager must be instanciated or if all watches
        must be removed and no other watches need to be added.
        """
        os.close(self._fd)

    def get_fd(self):
        """
        Return assigned inotify's file descriptor.

        @return: File descriptor.
        @rtype: int
        """
        return self._fd

    def get_watch(self, wd):
        """
        Get watch from provided watch descriptor wd.

        @param wd: Watch descriptor.
        @type wd: int
        """
        return self._wmd.get(wd)

    def del_watch(self, wd):
        """
        Remove watch entry associated to watch descriptor wd.

        @param wd: Watch descriptor.
        @type wd: int
        """
        try:
            del self._wmd[wd]
        except KeyError as err:
            log.error('Cannot delete unknown watch descriptor %s' % str(err))

    @property
    def watches(self):
        """
        Get a reference on the internal watch manager dictionary.

        @return: Internal watch manager dictionary.
        @rtype: dict
        """
        return self._wmd

    def __format_path(self, path):
        """
        Format path to its internal (stored in watch manager) representation.
        """
        return os.path.normpath(path)

    def __add_watch(self, path, mask, proc_fun, auto_add, exclude_filter):
        """
        Add a watch on path, build a Watch object and insert it in the
        watch manager dictionary. Return the wd value.
        """
        path = self.__format_path(path)
        if auto_add and (not mask & IN_CREATE):
            mask |= IN_CREATE
        wd = self._inotify_wrapper.inotify_add_watch(self._fd, path, mask)
        if wd < 0:
            return wd
        watch = Watch(wd=wd, path=path, mask=mask, proc_fun=proc_fun, auto_add=auto_add, exclude_filter=exclude_filter)
        self._wmd[wd] = watch
        log.debug('New %s', watch)
        return wd

    def __glob(self, path, do_glob):
        if do_glob:
            return glob.iglob(path)
        else:
            return [path]

    def add_watch(self, path, mask, proc_fun=None, rec=False, auto_add=False, do_glob=False, quiet=True, exclude_filter=None):
        """
        Add watch(s) on the provided |path|(s) with associated |mask| flag
        value and optionally with a processing |proc_fun| function and
        recursive flag |rec| set to True.
        All |path| components _must_ be str (i.e. unicode) objects.
        If |path| is already watched it is ignored, but if it is called with
        option rec=True a watch is put on each one of its not-watched
        subdirectory.

        @param path: Path to watch, the path can either be a file or a
                     directory. Also accepts a sequence (list) of paths.
        @type path: string or list of strings
        @param mask: Bitmask of events.
        @type mask: int
        @param proc_fun: Processing object.
        @type proc_fun: function or ProcessEvent instance or instance of
                        one of its subclasses or callable object.
        @param rec: Recursively add watches from path on all its
                    subdirectories, set to False by default (doesn't
                    follows symlinks in any case).
        @type rec: bool
        @param auto_add: Automatically add watches on newly created
                         directories in watched parent |path| directory.
                         If |auto_add| is True, IN_CREATE is ored with |mask|
                         when the watch is added.
        @type auto_add: bool
        @param do_glob: Do globbing on pathname (see standard globbing
                        module for more informations).
        @type do_glob: bool
        @param quiet: if False raises a WatchManagerError exception on
                      error. See example not_quiet.py.
        @type quiet: bool
        @param exclude_filter: predicate (boolean function), which returns
                               True if the current path must be excluded
                               from being watched. This argument has
                               precedence over exclude_filter passed to
                               the class' constructor.
        @type exclude_filter: callable object
        @return: dict of paths associated to watch descriptors. A wd value
                 is positive if the watch was added sucessfully, otherwise
                 the value is negative. If the path was invalid or was already
                 watched it is not included into this returned dictionary.
        @rtype: dict of {str: int}
        """
        ret_ = {}
        if exclude_filter is None:
            exclude_filter = self._exclude_filter
        for npath in self.__format_param(path):
            if not isinstance(npath, str):
                ret_[path] = -3
                continue
            for apath in self.__glob(npath, do_glob):
                for rpath in self.__walk_rec(apath, rec):
                    if not exclude_filter(rpath):
                        wd = ret_[rpath] = self.__add_watch(rpath, mask, proc_fun, auto_add, exclude_filter)
                        if wd < 0:
                            err = 'add_watch: cannot watch %s WD=%d, %s' % (rpath, wd, self._inotify_wrapper.str_errno())
                            if quiet:
                                log.error(err)
                            else:
                                raise WatchManagerError(err, ret_)
                    else:
                        ret_[rpath] = -2
        return ret_

    def __get_sub_rec(self, lpath):
        """
        Get every wd from self._wmd if its path is under the path of
        one (at least) of those in lpath. Doesn't follow symlinks.

        @param lpath: list of watch descriptor
        @type lpath: list of int
        @return: list of watch descriptor
        @rtype: list of int
        """
        for d in lpath:
            root = self.get_path(d)
            if root is not None:
                yield d
            else:
                continue
            if not os.path.isdir(root):
                continue
            root = os.path.normpath(root)
            lend = len(root)
            for iwd in self._wmd.items():
                cur = iwd[1].path
                pref = os.path.commonprefix([root, cur])
                if root == os.sep or (len(pref) == lend and len(cur) > lend and (cur[lend] == os.sep)):
                    yield iwd[1].wd

    def update_watch(self, wd, mask=None, proc_fun=None, rec=False, auto_add=False, quiet=True):
        """
        Update existing watch descriptors |wd|. The |mask| value, the
        processing object |proc_fun|, the recursive param |rec| and the
        |auto_add| and |quiet| flags can all be updated.

        @param wd: Watch Descriptor to update. Also accepts a list of
                   watch descriptors.
        @type wd: int or list of int
        @param mask: Optional new bitmask of events.
        @type mask: int
        @param proc_fun: Optional new processing function.
        @type proc_fun: function or ProcessEvent instance or instance of
                        one of its subclasses or callable object.
        @param rec: Optionally adds watches recursively on all
                    subdirectories contained into |wd| directory.
        @type rec: bool
        @param auto_add: Automatically adds watches on newly created
                         directories in the watch's path corresponding to |wd|.
                         If |auto_add| is True, IN_CREATE is ored with |mask|
                         when the watch is updated.
        @type auto_add: bool
        @param quiet: If False raises a WatchManagerError exception on
                      error. See example not_quiet.py
        @type quiet: bool
        @return: dict of watch descriptors associated to booleans values.
                 True if the corresponding wd has been successfully
                 updated, False otherwise.
        @rtype: dict of {int: bool}
        """
        lwd = self.__format_param(wd)
        if rec:
            lwd = self.__get_sub_rec(lwd)
        ret_ = {}
        for awd in lwd:
            apath = self.get_path(awd)
            if not apath or awd < 0:
                err = 'update_watch: invalid WD=%d' % awd
                if quiet:
                    log.error(err)
                    continue
                raise WatchManagerError(err, ret_)
            if mask:
                wd_ = self._inotify_wrapper.inotify_add_watch(self._fd, apath, mask)
                if wd_ < 0:
                    ret_[awd] = False
                    err = 'update_watch: cannot update %s WD=%d, %s' % (apath, wd_, self._inotify_wrapper.str_errno())
                    if quiet:
                        log.error(err)
                        continue
                    raise WatchManagerError(err, ret_)
                assert awd == wd_
            if proc_fun or auto_add:
                watch_ = self._wmd[awd]
            if proc_fun:
                watch_.proc_fun = proc_fun
            if auto_add:
                watch_.auto_add = auto_add
            ret_[awd] = True
            log.debug('Updated watch - %s', self._wmd[awd])
        return ret_

    def __format_param(self, param):
        """
        @param param: Parameter.
        @type param: string or int
        @return: wrap param.
        @rtype: list of type(param)
        """
        if isinstance(param, list):
            for p_ in param:
                yield p_
        else:
            yield param

    def get_wd(self, path):
        """
        Returns the watch descriptor associated to path. This method
        presents a prohibitive cost, always prefer to keep the WD
        returned by add_watch(). If the path is unknown it returns None.

        @param path: Path.
        @type path: str
        @return: WD or None.
        @rtype: int or None
        """
        path = self.__format_path(path)
        for iwd in self._wmd.items():
            if iwd[1].path == path:
                return iwd[0]

    def get_path(self, wd):
        """
        Returns the path associated to WD, if WD is unknown it returns None.

        @param wd: Watch descriptor.
        @type wd: int
        @return: Path or None.
        @rtype: string or None
        """
        watch_ = self._wmd.get(wd)
        if watch_ is not None:
            return watch_.path

    def __walk_rec(self, top, rec):
        """
        Yields each subdirectories of top, doesn't follow symlinks.
        If rec is false, only yield top.

        @param top: root directory.
        @type top: string
        @param rec: recursive flag.
        @type rec: bool
        @return: path of one subdirectory.
        @rtype: string
        """
        if not rec or os.path.islink(top) or (not os.path.isdir(top)):
            yield top
        else:
            for root, dirs, files in os.walk(top):
                yield root

    def rm_watch(self, wd, rec=False, quiet=True):
        """
        Removes watch(s).

        @param wd: Watch Descriptor of the file or directory to unwatch.
                   Also accepts a list of WDs.
        @type wd: int or list of int.
        @param rec: Recursively removes watches on every already watched
                    subdirectories and subfiles.
        @type rec: bool
        @param quiet: If False raises a WatchManagerError exception on
                      error. See example not_quiet.py
        @type quiet: bool
        @return: dict of watch descriptors associated to booleans values.
                 True if the corresponding wd has been successfully
                 removed, False otherwise.
        @rtype: dict of {int: bool}
        """
        lwd = self.__format_param(wd)
        if rec:
            lwd = self.__get_sub_rec(lwd)
        ret_ = {}
        for awd in lwd:
            wd_ = self._inotify_wrapper.inotify_rm_watch(self._fd, awd)
            if wd_ < 0:
                ret_[awd] = False
                err = 'rm_watch: cannot remove WD=%d, %s' % (awd, self._inotify_wrapper.str_errno())
                if quiet:
                    log.error(err)
                    continue
                raise WatchManagerError(err, ret_)
            if awd in self._wmd:
                del self._wmd[awd]
            ret_[awd] = True
            log.debug('Watch WD=%d (%s) removed', awd, self.get_path(awd))
        return ret_

    def watch_transient_file(self, filename, mask, proc_class):
        """
        Watch a transient file, which will be created and deleted frequently
        over time (e.g. pid file).

        @attention: Currently under the call to this function it is not
        possible to correctly watch the events triggered into the same
        base directory than the directory where is located this watched
        transient file. For instance it would be wrong to make these
        two successive calls: wm.watch_transient_file('/var/run/foo.pid', ...)
        and wm.add_watch('/var/run/', ...)

        @param filename: Filename.
        @type filename: string
        @param mask: Bitmask of events, should contain IN_CREATE and IN_DELETE.
        @type mask: int
        @param proc_class: ProcessEvent (or of one of its subclass), beware of
                           accepting a ProcessEvent's instance as argument into
                           __init__, see transient_file.py example for more
                           details.
        @type proc_class: ProcessEvent's instance or of one of its subclasses.
        @return: Same as add_watch().
        @rtype: Same as add_watch().
        """
        dirname = os.path.dirname(filename)
        if dirname == '':
            return {}
        basename = os.path.basename(filename)
        mask |= IN_CREATE | IN_DELETE

        def cmp_name(event):
            if getattr(event, 'name') is None:
                return False
            return basename == event.name
        return self.add_watch(dirname, mask, proc_fun=proc_class(ChainIfTrue(func=cmp_name)), rec=False, auto_add=False, do_glob=False, exclude_filter=lambda path: False)

    def get_ignore_events(self):
        return self._ignore_events

    def set_ignore_events(self, nval):
        self._ignore_events = nval
    ignore_events = property(get_ignore_events, set_ignore_events, 'Make watch manager ignoring new events.')