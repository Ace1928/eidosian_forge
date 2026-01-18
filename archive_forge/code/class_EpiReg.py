import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EpiReg(FSLCommand):
    """

    Runs FSL epi_reg script for simultaneous coregistration and fieldmap
    unwarping.

    Examples
    --------

    >>> from nipype.interfaces.fsl import EpiReg
    >>> epireg = EpiReg()
    >>> epireg.inputs.epi='epi.nii'
    >>> epireg.inputs.t1_head='T1.nii'
    >>> epireg.inputs.t1_brain='T1_brain.nii'
    >>> epireg.inputs.out_base='epi2struct'
    >>> epireg.inputs.fmap='fieldmap_phase_fslprepared.nii'
    >>> epireg.inputs.fmapmag='fieldmap_mag.nii'
    >>> epireg.inputs.fmapmagbrain='fieldmap_mag_brain.nii'
    >>> epireg.inputs.echospacing=0.00067
    >>> epireg.inputs.pedir='y'
    >>> epireg.cmdline # doctest: +ELLIPSIS
    'epi_reg --echospacing=0.000670 --fmap=fieldmap_phase_fslprepared.nii --fmapmag=fieldmap_mag.nii --fmapmagbrain=fieldmap_mag_brain.nii --noclean --pedir=y --epi=epi.nii --t1=T1.nii --t1brain=T1_brain.nii --out=epi2struct'
    >>> epireg.run() # doctest: +SKIP

    """
    _cmd = 'epi_reg'
    input_spec = EpiRegInputSpec
    output_spec = EpiRegOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.join(os.getcwd(), self.inputs.out_base + '.nii.gz')
        if not (isdefined(self.inputs.no_fmapreg) and self.inputs.no_fmapreg) and isdefined(self.inputs.fmap):
            outputs['out_1vol'] = os.path.join(os.getcwd(), self.inputs.out_base + '_1vol.nii.gz')
            outputs['fmap2str_mat'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fieldmap2str.mat')
            outputs['fmap2epi_mat'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fieldmaprads2epi.mat')
            outputs['fmap_epi'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fieldmaprads2epi.nii.gz')
            outputs['fmap_str'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fieldmaprads2str.nii.gz')
            outputs['fmapmag_str'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fieldmap2str.nii.gz')
            outputs['shiftmap'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fieldmaprads2epi_shift.nii.gz')
            outputs['fullwarp'] = os.path.join(os.getcwd(), self.inputs.out_base + '_warp.nii.gz')
            outputs['epi2str_inv'] = os.path.join(os.getcwd(), self.inputs.out_base + '_inv.mat')
        if not isdefined(self.inputs.wmseg):
            outputs['wmedge'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fast_wmedge.nii.gz')
            outputs['wmseg'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fast_wmseg.nii.gz')
            outputs['seg'] = os.path.join(os.getcwd(), self.inputs.out_base + '_fast_seg.nii.gz')
        outputs['epi2str_mat'] = os.path.join(os.getcwd(), self.inputs.out_base + '.mat')
        return outputs