import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FUGUE(FSLCommand):
    """FSL FUGUE set of tools for EPI distortion correction

    `FUGUE <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE>`_ is, most generally,
    a set of tools for EPI distortion correction.

    Distortions may be corrected for
        1. improving registration with non-distorted images (e.g. structurals),
           or
        2. dealing with motion-dependent changes.

    FUGUE is designed to deal only with the first case -
    improving registration.


    Examples
    --------


    Unwarping an input image (shift map is known):

    >>> from nipype.interfaces.fsl.preprocess import FUGUE
    >>> fugue = FUGUE()
    >>> fugue.inputs.in_file = 'epi.nii'
    >>> fugue.inputs.mask_file = 'epi_mask.nii'
    >>> fugue.inputs.shift_in_file = 'vsm.nii'  # Previously computed with fugue as well
    >>> fugue.inputs.unwarp_direction = 'y'
    >>> fugue.inputs.output_type = "NIFTI_GZ"
    >>> fugue.cmdline # doctest: +ELLIPSIS
    'fugue --in=epi.nii --mask=epi_mask.nii --loadshift=vsm.nii --unwarpdir=y --unwarp=epi_unwarped.nii.gz'
    >>> fugue.run() #doctest: +SKIP


    Warping an input image (shift map is known):

    >>> from nipype.interfaces.fsl.preprocess import FUGUE
    >>> fugue = FUGUE()
    >>> fugue.inputs.in_file = 'epi.nii'
    >>> fugue.inputs.forward_warping = True
    >>> fugue.inputs.mask_file = 'epi_mask.nii'
    >>> fugue.inputs.shift_in_file = 'vsm.nii'  # Previously computed with fugue as well
    >>> fugue.inputs.unwarp_direction = 'y'
    >>> fugue.inputs.output_type = "NIFTI_GZ"
    >>> fugue.cmdline # doctest: +ELLIPSIS
    'fugue --in=epi.nii --mask=epi_mask.nii --loadshift=vsm.nii --unwarpdir=y --warp=epi_warped.nii.gz'
    >>> fugue.run() #doctest: +SKIP


    Computing the vsm (unwrapped phase map is known):

    >>> from nipype.interfaces.fsl.preprocess import FUGUE
    >>> fugue = FUGUE()
    >>> fugue.inputs.phasemap_in_file = 'epi_phasediff.nii'
    >>> fugue.inputs.mask_file = 'epi_mask.nii'
    >>> fugue.inputs.dwell_to_asym_ratio = (0.77e-3 * 3) / 2.46e-3
    >>> fugue.inputs.unwarp_direction = 'y'
    >>> fugue.inputs.save_shift = True
    >>> fugue.inputs.output_type = "NIFTI_GZ"
    >>> fugue.cmdline # doctest: +ELLIPSIS
    'fugue --dwelltoasym=0.9390243902 --mask=epi_mask.nii --phasemap=epi_phasediff.nii --saveshift=epi_phasediff_vsm.nii.gz --unwarpdir=y'
    >>> fugue.run() #doctest: +SKIP


    """
    _cmd = 'fugue'
    input_spec = FUGUEInputSpec
    output_spec = FUGUEOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        input_phase = isdefined(self.inputs.phasemap_in_file)
        input_vsm = isdefined(self.inputs.shift_in_file)
        input_fmap = isdefined(self.inputs.fmap_in_file)
        if not input_phase and (not input_vsm) and (not input_fmap):
            raise RuntimeError('Either phasemap_in_file, shift_in_file or fmap_in_file must be set.')
        if not isdefined(self.inputs.in_file):
            skip += ['unwarped_file', 'warped_file']
        elif self.inputs.forward_warping:
            skip += ['unwarped_file']
            trait_spec = self.inputs.trait('warped_file')
            trait_spec.name_template = '%s_warped'
            trait_spec.name_source = 'in_file'
            trait_spec.output_name = 'warped_file'
        else:
            skip += ['warped_file']
            trait_spec = self.inputs.trait('unwarped_file')
            trait_spec.name_template = '%s_unwarped'
            trait_spec.name_source = 'in_file'
            trait_spec.output_name = 'unwarped_file'
        if not isdefined(self.inputs.shift_out_file):
            vsm_save_masked = isdefined(self.inputs.save_shift) and self.inputs.save_shift
            vsm_save_unmasked = isdefined(self.inputs.save_unmasked_shift) and self.inputs.save_unmasked_shift
            if vsm_save_masked or vsm_save_unmasked:
                trait_spec = self.inputs.trait('shift_out_file')
                trait_spec.output_name = 'shift_out_file'
                if input_fmap:
                    trait_spec.name_source = 'fmap_in_file'
                elif input_phase:
                    trait_spec.name_source = 'phasemap_in_file'
                elif input_vsm:
                    trait_spec.name_source = 'shift_in_file'
                else:
                    raise RuntimeError('Either phasemap_in_file, shift_in_file or fmap_in_file must be set.')
                if vsm_save_unmasked:
                    trait_spec.name_template = '%s_vsm_unmasked'
                else:
                    trait_spec.name_template = '%s_vsm'
            else:
                skip += ['save_shift', 'save_unmasked_shift', 'shift_out_file']
        if not isdefined(self.inputs.fmap_out_file):
            fmap_save_masked = isdefined(self.inputs.save_fmap) and self.inputs.save_fmap
            fmap_save_unmasked = isdefined(self.inputs.save_unmasked_fmap) and self.inputs.save_unmasked_fmap
            if fmap_save_masked or fmap_save_unmasked:
                trait_spec = self.inputs.trait('fmap_out_file')
                trait_spec.output_name = 'fmap_out_file'
                if input_vsm:
                    trait_spec.name_source = 'shift_in_file'
                elif input_phase:
                    trait_spec.name_source = 'phasemap_in_file'
                elif input_fmap:
                    trait_spec.name_source = 'fmap_in_file'
                else:
                    raise RuntimeError('Either phasemap_in_file, shift_in_file or fmap_in_file must be set.')
                if fmap_save_unmasked:
                    trait_spec.name_template = '%s_fieldmap_unmasked'
                else:
                    trait_spec.name_template = '%s_fieldmap'
            else:
                skip += ['save_fmap', 'save_unmasked_fmap', 'fmap_out_file']
        return super(FUGUE, self)._parse_inputs(skip=skip)