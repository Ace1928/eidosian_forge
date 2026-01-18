import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EpiRegInputSpec(FSLCommandInputSpec):
    epi = File(exists=True, argstr='--epi=%s', mandatory=True, position=-4, desc='EPI image')
    t1_head = File(exists=True, argstr='--t1=%s', mandatory=True, position=-3, desc='wholehead T1 image')
    t1_brain = File(exists=True, argstr='--t1brain=%s', mandatory=True, position=-2, desc='brain extracted T1 image')
    out_base = traits.String('epi2struct', desc='output base name', argstr='--out=%s', position=-1, usedefault=True)
    fmap = File(exists=True, argstr='--fmap=%s', desc='fieldmap image (in rad/s)')
    fmapmag = File(exists=True, argstr='--fmapmag=%s', desc='fieldmap magnitude image - wholehead')
    fmapmagbrain = File(exists=True, argstr='--fmapmagbrain=%s', desc='fieldmap magnitude image - brain extracted')
    wmseg = File(exists=True, argstr='--wmseg=%s', desc='white matter segmentation of T1 image, has to be named                  like the t1brain and end on _wmseg')
    echospacing = traits.Float(argstr='--echospacing=%f', desc='Effective EPI echo spacing                                 (sometimes called dwell time) - in seconds')
    pedir = traits.Enum('x', 'y', 'z', '-x', '-y', '-z', argstr='--pedir=%s', desc='phase encoding direction, dir = x/y/z/-x/-y/-z')
    weight_image = File(exists=True, argstr='--weight=%s', desc='weighting image (in T1 space)')
    no_fmapreg = traits.Bool(False, argstr='--nofmapreg', desc='do not perform registration of fmap to T1                         (use if fmap already registered)')
    no_clean = traits.Bool(True, argstr='--noclean', usedefault=True, desc='do not clean up intermediate files')