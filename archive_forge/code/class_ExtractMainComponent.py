import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ExtractMainComponent(CommandLine):
    """Extract the main component of a tessellated surface

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import ExtractMainComponent
    >>> mcmp = ExtractMainComponent(in_file='lh.pial')
    >>> mcmp.cmdline
    'mris_extract_main_component lh.pial lh.maincmp'

    """
    _cmd = 'mris_extract_main_component'
    input_spec = ExtractMainComponentInputSpec
    output_spec = ExtractMainComponentOutputSpec