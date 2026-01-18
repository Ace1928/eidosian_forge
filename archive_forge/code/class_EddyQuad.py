import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyQuad(FSLCommand):
    """
    Interface for FSL eddy_quad, a tool for generating single subject reports
    and storing the quality assessment indices for each subject.
    `User guide <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddyqc/UsersGuide>`__

    Examples
    --------

    >>> from nipype.interfaces.fsl import EddyQuad
    >>> quad = EddyQuad()
    >>> quad.inputs.base_name  = 'eddy_corrected'
    >>> quad.inputs.idx_file   = 'epi_index.txt'
    >>> quad.inputs.param_file = 'epi_acqp.txt'
    >>> quad.inputs.mask_file  = 'epi_mask.nii'
    >>> quad.inputs.bval_file  = 'bvals.scheme'
    >>> quad.inputs.bvec_file  = 'bvecs.scheme'
    >>> quad.inputs.output_dir = 'eddy_corrected.qc'
    >>> quad.inputs.field      = 'fieldmap_phase_fslprepared.nii'
    >>> quad.inputs.verbose    = True
    >>> quad.cmdline
    'eddy_quad eddy_corrected --bvals bvals.scheme --bvecs bvecs.scheme --field fieldmap_phase_fslprepared.nii --eddyIdx epi_index.txt --mask epi_mask.nii --output-dir eddy_corrected.qc --eddyParams epi_acqp.txt --verbose'
    >>> res = quad.run() # doctest: +SKIP

    """
    _cmd = 'eddy_quad'
    input_spec = EddyQuadInputSpec
    output_spec = EddyQuadOutputSpec

    def _list_outputs(self):
        from glob import glob
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.output_dir):
            out_dir = os.path.abspath(os.path.basename(self.inputs.base_name) + '.qc')
        else:
            out_dir = os.path.abspath(self.inputs.output_dir)
        outputs['qc_json'] = os.path.join(out_dir, 'qc.json')
        outputs['qc_pdf'] = os.path.join(out_dir, 'qc.pdf')
        outputs['avg_b_png'] = sorted(glob(os.path.join(out_dir, 'avg_b*.png')))
        if isdefined(self.inputs.field):
            outputs['avg_b0_pe_png'] = sorted(glob(os.path.join(out_dir, 'avg_b0_pe*.png')))
            for fname in outputs['avg_b0_pe_png']:
                outputs['avg_b_png'].remove(fname)
            outputs['vdm_png'] = os.path.join(out_dir, 'vdm.png')
        outputs['cnr_png'] = sorted(glob(os.path.join(out_dir, 'cnr*.png')))
        residuals = os.path.join(out_dir, 'eddy_msr.txt')
        if os.path.isfile(residuals):
            outputs['residuals'] = residuals
        clean_volumes = os.path.join(out_dir, 'vols_no_outliers.txt')
        if os.path.isfile(clean_volumes):
            outputs['clean_volumes'] = clean_volumes
        return outputs