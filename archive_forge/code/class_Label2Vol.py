import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2Vol(FSCommand):
    """Make a binary volume from a Freesurfer label

    Examples
    --------
    >>> binvol = Label2Vol(label_file='cortex.label', template_file='structural.nii', reg_file='register.dat', fill_thresh=0.5, vol_label_file='foo_out.nii')
    >>> binvol.cmdline
    'mri_label2vol --fillthresh 0.5 --label cortex.label --reg register.dat --temp structural.nii --o foo_out.nii'

    """
    _cmd = 'mri_label2vol'
    input_spec = Label2VolInputSpec
    output_spec = Label2VolOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outfile = self.inputs.vol_label_file
        if not isdefined(outfile):
            for key in ['label_file', 'annot_file', 'seg_file']:
                if isdefined(getattr(self.inputs, key)):
                    path = getattr(self.inputs, key)
                    if isinstance(path, list):
                        path = path[0]
                    _, src = os.path.split(path)
            if isdefined(self.inputs.aparc_aseg):
                src = 'aparc+aseg.mgz'
            outfile = fname_presuffix(src, suffix='_vol.nii.gz', newpath=os.getcwd(), use_ext=False)
        outputs['vol_label_file'] = outfile
        return outputs

    def _gen_filename(self, name):
        if name == 'vol_label_file':
            return self._list_outputs()[name]
        return None