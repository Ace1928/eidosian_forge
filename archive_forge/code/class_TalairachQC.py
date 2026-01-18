import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class TalairachQC(FSScriptCommand):
    """
    Examples
    ========

    >>> from nipype.interfaces.freesurfer import TalairachQC
    >>> qc = TalairachQC()
    >>> qc.inputs.log_file = 'dirs.txt'
    >>> qc.cmdline
    'tal_QC_AZS dirs.txt'
    """
    _cmd = 'tal_QC_AZS'
    input_spec = TalairachQCInputSpec
    output_spec = FSScriptOutputSpec