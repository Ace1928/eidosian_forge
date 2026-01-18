import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
class AFNICommandBase(CommandLine):
    """
    A base class to fix a linking problem in OSX and AFNI.

    See Also
    --------
    `This thread
    <http://afni.nimh.nih.gov/afni/community/board/read.php?1,145346,145347#msg-145347>`__
    about the particular environment variable that fixes this problem.

    """

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        if platform == 'darwin':
            runtime.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/usr/local/afni/'
        return super(AFNICommandBase, self)._run_interface(runtime, correct_return_codes)