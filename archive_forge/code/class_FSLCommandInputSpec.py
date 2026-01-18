from glob import glob
import os
from ... import logging
from ...utils.filemanip import fname_presuffix
from ..base import traits, isdefined, CommandLine, CommandLineInputSpec, PackageInfo
from ...external.due import BibTeX
class FSLCommandInputSpec(CommandLineInputSpec):
    """
    Base Input Specification for all FSL Commands

    All command support specifying FSLOUTPUTTYPE dynamically
    via output_type.

    Example
    -------
    fsl.ExtractRoi(tmin=42, tsize=1, output_type='NIFTI')
    """
    output_type = traits.Enum('NIFTI', list(Info.ftypes.keys()), desc='FSL output type')