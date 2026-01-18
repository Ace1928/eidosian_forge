import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.interfaces.cat12.base import Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import (
from nipype.utils.filemanip import split_filename, fname_presuffix
class CAT12SegmentOutputSpec(TraitedSpec):
    label_files = traits.List(File(exists=True), desc='Files with the measures extracted for OI ands ROIs')
    label_rois = File(exists=True, desc='Files with thickness values of ROIs.')
    label_roi = File(exists=True, desc='Files with thickness values of ROI.')
    mri_images = traits.List(File(exists=True), desc='Different segmented images.')
    gm_modulated_image = File(exists=True, desc='Grey matter modulated image.')
    gm_dartel_image = File(exists=True, desc='Grey matter dartel image.')
    gm_native_image = File(exists=True, desc='Grey matter native space.')
    wm_modulated_image = File(exists=True, desc='White matter modulated image.')
    wm_dartel_image = File(exists=True, desc='White matter dartel image.')
    wm_native_image = File(exists=True, desc='White matter in native space.')
    csf_modulated_image = File(exists=True, desc='CSF modulated image.')
    csf_dartel_image = File(exists=True, desc='CSF dartel image.')
    csf_native_image = File(exists=True, desc='CSF in native space.')
    bias_corrected_image = File(exists=True, desc='Bias corrected image')
    surface_files = traits.List(File(exists=True), desc='Surface files')
    rh_central_surface = File(exists=True, desc='Central right hemisphere files')
    rh_sphere_surface = File(exists=True, desc='Sphere right hemisphere files')
    lh_central_surface = File(exists=True, desc='Central left hemisphere files')
    lh_sphere_surface = File(exists=True, desc='Sphere left hemisphere files')
    report_files = traits.List(File(exists=True), desc='Report files.')
    report = File(exists=True, desc='Report file.')