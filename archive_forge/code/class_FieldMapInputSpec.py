import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class FieldMapInputSpec(SPMCommandInputSpec):
    jobtype = traits.Enum('calculatevdm', usedefault=True, deprecated='1.9.0', desc="Must be 'calculatevdm'; to apply VDM, use the ApplyVDM interface.")
    phase_file = File(mandatory=True, exists=True, copyfile=False, field='subj.data.presubphasemag.phase', desc='presubstracted phase file')
    magnitude_file = File(mandatory=True, exists=True, copyfile=False, field='subj.data.presubphasemag.magnitude', desc='presubstracted magnitude file')
    echo_times = traits.Tuple(traits.Float, traits.Float, mandatory=True, field='subj.defaults.defaultsval.et', desc='short and long echo times')
    maskbrain = traits.Bool(True, usedefault=True, field='subj.defaults.defaultsval.maskbrain', desc='masking or no masking of the brain')
    blip_direction = traits.Enum(1, -1, mandatory=True, field='subj.defaults.defaultsval.blipdir', desc='polarity of the phase-encode blips')
    total_readout_time = traits.Float(mandatory=True, field='subj.defaults.defaultsval.tert', desc='total EPI readout time')
    epifm = traits.Bool(False, usedefault=True, field='subj.defaults.defaultsval.epifm', desc='epi-based field map')
    jacobian_modulation = traits.Bool(False, usedefault=True, field='subj.defaults.defaultsval.ajm', desc='jacobian modulation')
    method = traits.Enum('Mark3D', 'Mark2D', 'Huttonish', usedefault=True, desc='One of: Mark3D, Mark2D, Huttonish', field='subj.defaults.defaultsval.uflags.method')
    unwarp_fwhm = traits.Range(low=0, value=10, usedefault=True, field='subj.defaults.defaultsval.uflags.fwhm', desc='gaussian smoothing kernel width')
    pad = traits.Range(low=0, value=0, usedefault=True, field='subj.defaults.defaultsval.uflags.pad', desc='padding kernel width')
    ws = traits.Bool(True, usedefault=True, field='subj.defaults.defaultsval.uflags.ws', desc='weighted smoothing')
    template = File(copyfile=False, exists=True, field='subj.defaults.defaultsval.mflags.template', desc='template image for brain masking')
    mask_fwhm = traits.Range(low=0, value=5, usedefault=True, field='subj.defaults.defaultsval.mflags.fwhm', desc='gaussian smoothing kernel width')
    nerode = traits.Range(low=0, value=2, usedefault=True, field='subj.defaults.defaultsval.mflags.nerode', desc='number of erosions')
    ndilate = traits.Range(low=0, value=4, usedefault=True, field='subj.defaults.defaultsval.mflags.ndilate', desc='number of erosions')
    thresh = traits.Float(0.5, usedefault=True, field='subj.defaults.defaultsval.mflags.thresh', desc='threshold used to create brain mask from segmented data')
    reg = traits.Float(0.02, usedefault=True, field='subj.defaults.defaultsval.mflags.reg', desc='regularization value used in the segmentation')
    epi_file = File(copyfile=False, exists=True, mandatory=True, field='subj.session.epi', desc='EPI to unwarp')
    matchvdm = traits.Bool(True, usedefault=True, field='subj.matchvdm', desc='match VDM to EPI')
    sessname = Str('_run-', usedefault=True, field='subj.sessname', desc='VDM filename extension')
    writeunwarped = traits.Bool(False, usedefault=True, field='subj.writeunwarped', desc='write unwarped EPI')
    anat_file = File(copyfile=False, exists=True, field='subj.anat', desc='anatomical image for comparison')
    matchanat = traits.Bool(True, usedefault=True, field='subj.matchanat', desc='match anatomical image to EPI')