import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class MRICoreg(FSCommand):
    """This program registers one volume to another

    mri_coreg is a C reimplementation of spm_coreg in FreeSurfer

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import MRICoreg
    >>> coreg = MRICoreg()
    >>> coreg.inputs.source_file = 'moving1.nii'
    >>> coreg.inputs.reference_file = 'fixed1.nii'
    >>> coreg.inputs.subjects_dir = '.'
    >>> coreg.cmdline # doctest: +ELLIPSIS
    'mri_coreg --lta .../registration.lta --ref fixed1.nii --mov moving1.nii --sd .'

    If passing a subject ID, the reference mask may be disabled:

    >>> coreg = MRICoreg()
    >>> coreg.inputs.source_file = 'moving1.nii'
    >>> coreg.inputs.subjects_dir = '.'
    >>> coreg.inputs.subject_id = 'fsaverage'
    >>> coreg.inputs.reference_mask = False
    >>> coreg.cmdline # doctest: +ELLIPSIS
    'mri_coreg --s fsaverage --no-ref-mask --lta .../registration.lta --mov moving1.nii --sd .'

    Spatial scales may be specified as a list of one or two separations:

    >>> coreg.inputs.sep = [4]
    >>> coreg.cmdline # doctest: +ELLIPSIS
    'mri_coreg --s fsaverage --no-ref-mask --lta .../registration.lta --sep 4 --mov moving1.nii --sd .'

    >>> coreg.inputs.sep = [4, 5]
    >>> coreg.cmdline # doctest: +ELLIPSIS
    'mri_coreg --s fsaverage --no-ref-mask --lta .../registration.lta --sep 4 --sep 5 --mov moving1.nii --sd .'
    """
    _cmd = 'mri_coreg'
    input_spec = MRICoregInputSpec
    output_spec = MRICoregOutputSpec

    def _format_arg(self, opt, spec, val):
        if opt in ('out_reg_file', 'out_lta_file', 'out_params_file') and val is True:
            val = self._list_outputs()[opt]
        elif opt == 'reference_mask' and val is False:
            return '--no-ref-mask'
        return super(MRICoreg, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_lta_file = self.inputs.out_lta_file
        if isdefined(out_lta_file):
            if out_lta_file is True:
                out_lta_file = 'registration.lta'
            outputs['out_lta_file'] = os.path.abspath(out_lta_file)
        out_reg_file = self.inputs.out_reg_file
        if isdefined(out_reg_file):
            if out_reg_file is True:
                out_reg_file = 'registration.dat'
            outputs['out_reg_file'] = os.path.abspath(out_reg_file)
        out_params_file = self.inputs.out_params_file
        if isdefined(out_params_file):
            if out_params_file is True:
                out_params_file = 'registration.par'
            outputs['out_params_file'] = os.path.abspath(out_params_file)
        return outputs