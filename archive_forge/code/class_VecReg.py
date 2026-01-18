import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class VecReg(FSLCommand):
    """Use FSL vecreg for registering vector data
    For complete details, see the FDT Documentation
    <http://www.fmrib.ox.ac.uk/fsl/fdt/fdt_vecreg.html>

    Example
    -------

    >>> from nipype.interfaces import fsl
    >>> vreg = fsl.VecReg(in_file='diffusion.nii',                  affine_mat='trans.mat',                  ref_vol='mni.nii',                  out_file='diffusion_vreg.nii')
    >>> vreg.cmdline
    'vecreg -t trans.mat -i diffusion.nii -o diffusion_vreg.nii -r mni.nii'

    """
    _cmd = 'vecreg'
    input_spec = VecRegInputSpec
    output_spec = VecRegOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.out_file):
            pth, base_name = os.path.split(self.inputs.in_file)
            self.inputs.out_file = self._gen_fname(base_name, cwd=os.path.abspath(pth), suffix='_vreg')
        return super(VecReg, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']) and isdefined(self.inputs.in_file):
            pth, base_name = os.path.split(self.inputs.in_file)
            outputs['out_file'] = self._gen_fname(base_name, cwd=os.path.abspath(pth), suffix='_vreg')
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        else:
            return None