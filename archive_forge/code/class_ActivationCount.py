import os
import nibabel as nb
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import split_filename
class ActivationCount(SimpleInterface):
    """
    Calculate a simple Activation Count Maps

    Adapted from: https://github.com/poldracklab/CNP_task_analysis/    blob/61c27f5992db9d8800884f8ffceb73e6957db8af/CNP_2nd_level_ACM.py
    """
    input_spec = ActivationCountInputSpec
    output_spec = ActivationCountOutputSpec

    def _run_interface(self, runtime):
        allmaps = nb.concat_images(self.inputs.in_files).dataobj
        acm_pos = np.mean(allmaps > self.inputs.threshold, axis=3, dtype=np.float32)
        acm_neg = np.mean(allmaps < -1.0 * self.inputs.threshold, axis=3, dtype=np.float32)
        acm_diff = acm_pos - acm_neg
        template_fname = self.inputs.in_files[0]
        ext = split_filename(template_fname)[2]
        fname_fmt = os.path.join(runtime.cwd, 'acm_{}' + ext).format
        self._results['out_file'] = fname_fmt('diff')
        self._results['acm_pos'] = fname_fmt('pos')
        self._results['acm_neg'] = fname_fmt('neg')
        img = nb.load(template_fname)
        img.__class__(acm_diff, img.affine, img.header).to_filename(self._results['out_file'])
        img.__class__(acm_pos, img.affine, img.header).to_filename(self._results['acm_pos'])
        img.__class__(acm_neg, img.affine, img.header).to_filename(self._results['acm_neg'])
        return runtime