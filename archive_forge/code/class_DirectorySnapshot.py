import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
class DirectorySnapshot(object):
    """
    A snapshot of stat information of files in a directory.

    :param path:
        The directory path for which a snapshot should be taken.
    :type path:
        ``str``
    :param recursive:
        ``True`` if the entire directory tree should be included in the
        snapshot; ``False`` otherwise.
    :type recursive:
        ``bool``
    :param walker_callback:
        .. deprecated:: 0.7.2
    :param stat:
        Use custom stat function that returns a stat structure for path.
        Currently only st_dev, st_ino, st_mode and st_mtime are needed.
        
        A function with the signature ``walker_callback(path, stat_info)``
        which will be called for every entry in the directory tree.
    :param listdir:
        Use custom listdir function. See ``os.listdir`` for details.
    """

    def __init__(self, path, recursive=True, walker_callback=lambda p, s: None, stat=default_stat, listdir=os.listdir):
        self._stat_info = {}
        self._inode_to_path = {}
        st = stat(path)
        self._stat_info[path] = st
        self._inode_to_path[st.st_ino, st.st_dev] = path

        def walk(root):
            try:
                paths = [os.path.join(root, name) for name in listdir(root)]
            except OSError as e:
                if e.errno == errno.ENOENT:
                    return
                else:
                    raise
            entries = []
            for p in paths:
                try:
                    entries.append((p, stat(p)))
                except OSError:
                    continue
            for _ in entries:
                yield _
            if recursive:
                for path, st in entries:
                    if S_ISDIR(st.st_mode):
                        for _ in walk(path):
                            yield _
        for p, st in walk(path):
            i = (st.st_ino, st.st_dev)
            self._inode_to_path[i] = p
            self._stat_info[p] = st
            walker_callback(p, st)

    @property
    def paths(self):
        """
        Set of file/directory paths in the snapshot.
        """
        return set(self._stat_info.keys())

    def path(self, id):
        """
        Returns path for id. None if id is unknown to this snapshot.
        """
        return self._inode_to_path.get(id)

    def inode(self, path):
        """ Returns an id for path. """
        st = self._stat_info[path]
        return (st.st_ino, st.st_dev)

    def isdir(self, path):
        return S_ISDIR(self._stat_info[path].st_mode)

    def mtime(self, path):
        return self._stat_info[path].st_mtime

    def stat_info(self, path):
        """
        Returns a stat information object for the specified path from
        the snapshot.

        Attached information is subject to change. Do not use unless
        you specify `stat` in constructor. Use :func:`inode`, :func:`mtime`,
        :func:`isdir` instead.

        :param path:
            The path for which stat information should be obtained
            from a snapshot.
        """
        return self._stat_info[path]

    def __sub__(self, previous_dirsnap):
        """Allow subtracting a DirectorySnapshot object instance from
        another.

        :returns:
            A :class:`DirectorySnapshotDiff` object.
        """
        return DirectorySnapshotDiff(previous_dirsnap, self)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self._stat_info)