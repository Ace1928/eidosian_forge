import os
import sys
import stat
import fnmatch
import collections
import errno
def _rmtree_safe_fd(topfd, path, onerror):
    try:
        with os.scandir(topfd) as scandir_it:
            entries = list(scandir_it)
    except OSError as err:
        err.filename = path
        onerror(os.scandir, path, sys.exc_info())
        return
    for entry in entries:
        fullname = os.path.join(path, entry.name)
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            is_dir = False
        else:
            if is_dir:
                try:
                    orig_st = entry.stat(follow_symlinks=False)
                    is_dir = stat.S_ISDIR(orig_st.st_mode)
                except OSError:
                    onerror(os.lstat, fullname, sys.exc_info())
                    continue
        if is_dir:
            try:
                dirfd = os.open(entry.name, os.O_RDONLY, dir_fd=topfd)
                dirfd_closed = False
            except OSError:
                onerror(os.open, fullname, sys.exc_info())
            else:
                try:
                    if os.path.samestat(orig_st, os.fstat(dirfd)):
                        _rmtree_safe_fd(dirfd, fullname, onerror)
                        try:
                            os.close(dirfd)
                        except OSError:
                            dirfd_closed = True
                            onerror(os.close, fullname, sys.exc_info())
                        dirfd_closed = True
                        try:
                            os.rmdir(entry.name, dir_fd=topfd)
                        except OSError:
                            onerror(os.rmdir, fullname, sys.exc_info())
                    else:
                        try:
                            raise OSError('Cannot call rmtree on a symbolic link')
                        except OSError:
                            onerror(os.path.islink, fullname, sys.exc_info())
                finally:
                    if not dirfd_closed:
                        try:
                            os.close(dirfd)
                        except OSError:
                            onerror(os.close, fullname, sys.exc_info())
        else:
            try:
                os.unlink(entry.name, dir_fd=topfd)
            except OSError:
                onerror(os.unlink, fullname, sys.exc_info())