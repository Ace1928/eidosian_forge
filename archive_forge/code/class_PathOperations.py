import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
class PathOperations(StaticClass):
    """
    Static methods for filename and pathname manipulation.
    """

    @staticmethod
    def path_is_relative(path):
        """
        @see: L{path_is_absolute}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  bool
        @return: C{True} if the path is relative, C{False} if it's absolute.
        """
        return win32.PathIsRelative(path)

    @staticmethod
    def path_is_absolute(path):
        """
        @see: L{path_is_relative}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  bool
        @return: C{True} if the path is absolute, C{False} if it's relative.
        """
        return not win32.PathIsRelative(path)

    @staticmethod
    def make_relative(path, current=None):
        """
        @type  path: str
        @param path: Absolute path.

        @type  current: str
        @param current: (Optional) Path to the current directory.

        @rtype:  str
        @return: Relative path.

        @raise WindowsError: It's impossible to make the path relative.
            This happens when the path and the current path are not on the
            same disk drive or network share.
        """
        return win32.PathRelativePathTo(pszFrom=current, pszTo=path)

    @staticmethod
    def make_absolute(path):
        """
        @type  path: str
        @param path: Relative path.

        @rtype:  str
        @return: Absolute path.
        """
        return win32.GetFullPathName(path)[0]

    @staticmethod
    def split_extension(pathname):
        """
        @type  pathname: str
        @param pathname: Absolute path.

        @rtype:  tuple( str, str )
        @return:
            Tuple containing the file and extension components of the filename.
        """
        filepart = win32.PathRemoveExtension(pathname)
        extpart = win32.PathFindExtension(pathname)
        return (filepart, extpart)

    @staticmethod
    def split_filename(pathname):
        """
        @type  pathname: str
        @param pathname: Absolute path.

        @rtype:  tuple( str, str )
        @return: Tuple containing the path to the file and the base filename.
        """
        filepart = win32.PathFindFileName(pathname)
        pathpart = win32.PathRemoveFileSpec(pathname)
        return (pathpart, filepart)

    @staticmethod
    def split_path(path):
        """
        @see: L{join_path}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  list( str... )
        @return: List of path components.
        """
        components = list()
        while path:
            next = win32.PathFindNextComponent(path)
            if next:
                prev = path[:-len(next)]
                components.append(prev)
            path = next
        return components

    @staticmethod
    def join_path(*components):
        """
        @see: L{split_path}

        @type  components: tuple( str... )
        @param components: Path components.

        @rtype:  str
        @return: Absolute or relative path.
        """
        if components:
            path = components[0]
            for next in components[1:]:
                path = win32.PathAppend(path, next)
        else:
            path = ''
        return path

    @staticmethod
    def native_to_win32_pathname(name):
        """
        @type  name: str
        @param name: Native (NT) absolute pathname.

        @rtype:  str
        @return: Win32 absolute pathname.
        """
        if name.startswith('\\'):
            if name.startswith('\\??\\'):
                name = name[4:]
            elif name.startswith('\\SystemRoot\\'):
                system_root_path = os.environ['SYSTEMROOT']
                if system_root_path.endswith('\\'):
                    system_root_path = system_root_path[:-1]
                name = system_root_path + name[11:]
            else:
                for drive_number in compat.xrange(ord('A'), ord('Z') + 1):
                    drive_letter = '%c:' % drive_number
                    try:
                        device_native_path = win32.QueryDosDevice(drive_letter)
                    except WindowsError:
                        e = sys.exc_info()[1]
                        if e.winerror in (win32.ERROR_FILE_NOT_FOUND, win32.ERROR_PATH_NOT_FOUND):
                            continue
                        raise
                    if not device_native_path.endswith('\\'):
                        device_native_path += '\\'
                    if name.startswith(device_native_path):
                        name = drive_letter + '\\' + name[len(device_native_path):]
                        break
        return name

    @staticmethod
    def pathname_to_filename(pathname):
        """
        Equivalent to: C{PathOperations.split_filename(pathname)[0]}

        @note: This function is preserved for backwards compatibility with
            WinAppDbg 1.4 and earlier. It may be removed in future versions.

        @type  pathname: str
        @param pathname: Absolute path to a file.

        @rtype:  str
        @return: Filename component of the path.
        """
        return win32.PathFindFileName(pathname)