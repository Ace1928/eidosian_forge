import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.interfaces.cat12.base import Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import (
from nipype.utils.filemanip import split_filename, fname_presuffix
class Cell2Str(Cell):

    def __str__(self):
        """Convert input to appropriate format for cat12"""
        return "{'%s'}" % self.to_string()