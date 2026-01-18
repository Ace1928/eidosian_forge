import os
import re
from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem
class FileParameter(Parameter):
    """
    Interfaces with the myriad of file options available from a QFileDialog.

    Note that the output can either be a single file string or list of files, depending on whether
    `fileMode='ExistingFiles'` is specified.

    Note that in all cases, absolute file paths are returned unless `relativeTo` is specified as
    elaborated below.

    ============== ========================================================
    **Options:**
    parent         Dialog parent
    winTitle       Title of dialog window
    nameFilter     File filter as required by the Qt dialog
    directory      Where in the file system to open this dialog
    selectFile     File to preselect
    relativeTo     Parent directory that, if provided, will be removed from the prefix of all returned paths. So,
                   if '/my/text/file.txt' was selected, and `relativeTo='my/text/'`, the return value would be
                   'file.txt'. This uses os.path.relpath under the hood, so expect that behavior.
    kwargs         Any enum value accepted by a QFileDialog and its value. Values can be a string or list of strings,
                   i.e. fileMode='AnyFile', options=['ShowDirsOnly', 'DontResolveSymlinks']
    ============== ========================================================
    """
    itemClass = FileParameterItem

    def __init__(self, **opts):
        opts.setdefault('readonly', True)
        super().__init__(**opts)