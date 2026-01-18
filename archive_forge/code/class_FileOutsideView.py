import re
from . import errors, osutils, transport
class FileOutsideView(errors.BzrError):
    _fmt = 'Specified file "%(file_name)s" is outside the current view: %(view_str)s'

    def __init__(self, file_name, view_files):
        self.file_name = file_name
        self.view_str = ', '.join(view_files)