import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
class AFNIPythonCommand(AFNICommand):
    """A subtype of AFNI command line for Python scripts."""

    @property
    def cmd(self):
        """Revise the command path."""
        orig_cmd = super(AFNIPythonCommand, self).cmd
        found = shutil.which(orig_cmd)
        return found if found is not None else orig_cmd

    @property
    def _cmd_prefix(self):
        return '{} '.format(self.inputs.py27_path)