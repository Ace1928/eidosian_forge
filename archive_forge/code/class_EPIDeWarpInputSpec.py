import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EPIDeWarpInputSpec(FSLCommandInputSpec):
    mag_file = File(exists=True, desc='Magnitude file', argstr='--mag %s', position=0, mandatory=True)
    dph_file = File(exists=True, desc='Phase file assumed to be scaled from 0 to 4095', argstr='--dph %s', mandatory=True)
    exf_file = File(exists=True, desc='example func volume (or use epi)', argstr='--exf %s')
    epi_file = File(exists=True, desc='EPI volume to unwarp', argstr='--epi %s')
    tediff = traits.Float(2.46, usedefault=True, desc='difference in B0 field map TEs', argstr='--tediff %s')
    esp = traits.Float(0.58, desc='EPI echo spacing', argstr='--esp %s', usedefault=True)
    sigma = traits.Int(2, usedefault=True, argstr='--sigma %s', desc='2D spatial gaussing smoothing                        stdev (default = 2mm)')
    vsm = traits.String(genfile=True, desc='voxel shift map', argstr='--vsm %s')
    exfdw = traits.String(desc='dewarped example func volume', genfile=True, argstr='--exfdw %s')
    epidw = traits.String(desc='dewarped epi volume', genfile=False, argstr='--epidw %s')
    tmpdir = traits.String(genfile=True, desc='tmpdir', argstr='--tmpdir %s')
    nocleanup = traits.Bool(True, usedefault=True, desc='no cleanup', argstr='--nocleanup')
    cleanup = traits.Bool(desc='cleanup', argstr='--cleanup')