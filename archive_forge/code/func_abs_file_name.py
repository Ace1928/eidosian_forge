import os
import os.path
import sys
def abs_file_name(dir_, file_name):
    """If 'file_name' starts with '/', returns a copy of 'file_name'.
    Otherwise, returns an absolute path to 'file_name' considering it relative
    to 'dir_', which itself must be absolute.  'dir_' may be None or the empty
    string, in which case the current working directory is used.

    Returns None if 'dir_' is None and getcwd() fails.

    This differs from os.path.abspath() in that it will never change the
    meaning of a file name.

    On Windows an absolute path contains ':' ( i.e: C:\\ ) """
    if file_name.startswith('/') or file_name.find(':') > -1:
        return file_name
    else:
        if dir_ is None or dir_ == '':
            try:
                dir_ = os.getcwd()
            except OSError:
                return None
        if dir_.endswith('/'):
            return dir_ + file_name
        else:
            return '%s/%s' % (dir_, file_name)