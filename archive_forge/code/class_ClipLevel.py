import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class ClipLevel(AFNICommandBase):
    """Estimates the value at which to clip the anatomical dataset so
       that background regions are set to zero.

    For complete details, see the `3dClipLevel Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dClipLevel.html>`_

    Examples
    --------
    >>> from nipype.interfaces.afni import preprocess
    >>> cliplevel = preprocess.ClipLevel()
    >>> cliplevel.inputs.in_file = 'anatomical.nii'
    >>> cliplevel.cmdline
    '3dClipLevel anatomical.nii'
    >>> res = cliplevel.run()  # doctest: +SKIP

    """
    _cmd = '3dClipLevel'
    input_spec = ClipLevelInputSpec
    output_spec = ClipLevelOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        outfile = os.path.join(os.getcwd(), 'stat_result.json')
        if runtime is None:
            try:
                clip_val = load_json(outfile)['stat']
            except IOError:
                return self.run().outputs
        else:
            clip_val = []
            for line in runtime.stdout.split('\n'):
                if line:
                    values = line.split()
                    if len(values) > 1:
                        clip_val.append([float(val) for val in values])
                    else:
                        clip_val.extend([float(val) for val in values])
            if len(clip_val) == 1:
                clip_val = clip_val[0]
            save_json(outfile, dict(stat=clip_val))
        outputs.clip_val = clip_val
        return outputs