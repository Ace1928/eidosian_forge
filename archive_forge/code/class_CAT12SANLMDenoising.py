import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.interfaces.cat12.base import Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import (
from nipype.utils.filemanip import split_filename, fname_presuffix
class CAT12SANLMDenoising(SPMCommand):
    """
    Spatially adaptive non-local means (SANLM) denoising filter

    This  function  applies  an spatial adaptive (sub-resolution) non-local means denoising filter
    to  the  data.  This  filter  will  remove  noise  while  preserving  edges. The filter strength is
    automatically estimated based on the standard deviation of the noise.

    This   filter   is  internally  used  in  the  segmentation  procedure  anyway.  Thus,  it  is  not
    necessary (and not recommended) to apply the filter before segmentation.
    ______________________________________________________________________
    Christian Gaser, Robert Dahnke
    Structural Brain Mapping Group (http://www.neuro.uni-jena.de)
    Departments of Neurology and Psychiatry
    Jena University Hospital
    ______________________________________________________________________

    Examples
    --------
    >>> from nipype.interfaces import cat12
    >>> c = cat12.CAT12SANLMDenoising()
    >>> c.inputs.in_files = 'anatomical.nii'
    >>> c.run() # doctest: +SKIP
    """
    input_spec = CAT12SANLMDenoisingInputSpec
    output_spec = CAT12SANLMDenoisingOutputSpec

    def __init__(self, **inputs):
        _local_version = SPMCommand().version
        if _local_version and '12.' in _local_version:
            self._jobtype = 'tools'
            self._jobname = 'cat.tools.sanlm'
        SPMCommand.__init__(self, **inputs)

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            if isinstance(val, list):
                return scans_for_fnames(val)
            else:
                return scans_for_fname(val)
        if opt == 'spm_type':
            type_map = {'same': 0, 'uint8': 2, 'uint16': 512, 'float32': 16}
            val = type_map[val]
        return super(CAT12SANLMDenoising, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = fname_presuffix(self.inputs.in_files[0], newpath=os.getcwd(), prefix=self.inputs.filename_prefix, suffix=self.inputs.filename_suffix)
        return outputs