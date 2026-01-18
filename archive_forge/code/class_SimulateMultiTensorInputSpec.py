from multiprocessing import Pool, cpu_count
import os.path as op
import numpy as np
import nibabel as nb
from ... import logging
from ..base import (
from .base import DipyBaseInterface
class SimulateMultiTensorInputSpec(BaseInterfaceInputSpec):
    in_dirs = InputMultiPath(File(exists=True), mandatory=True, desc='list of fibers (principal directions)')
    in_frac = InputMultiPath(File(exists=True), mandatory=True, desc='volume fraction of each fiber')
    in_vfms = InputMultiPath(File(exists=True), mandatory=True, desc='volume fractions of isotropic compartiments')
    in_mask = File(exists=True, desc='mask to simulate data')
    diff_iso = traits.List([0.003, 0.00096, 0.00068], traits.Float, usedefault=True, desc='Diffusivity of isotropic compartments')
    diff_sf = traits.Tuple((0.0017, 0.0002, 0.0002), traits.Float, traits.Float, traits.Float, usedefault=True, desc='Single fiber tensor')
    n_proc = traits.Int(0, usedefault=True, desc='number of processes')
    baseline = File(exists=True, mandatory=True, desc='baseline T2 signal')
    gradients = File(exists=True, desc='gradients file')
    in_bvec = File(exists=True, desc='input bvecs file')
    in_bval = File(exists=True, desc='input bvals file')
    num_dirs = traits.Int(32, usedefault=True, desc='number of gradient directions (when table is automatically generated)')
    bvalues = traits.List(traits.Int, value=[1000, 3000], usedefault=True, desc='list of b-values (when table is automatically generated)')
    out_file = File('sim_dwi.nii.gz', usedefault=True, desc='output file with fractions to be simluated')
    out_mask = File('sim_msk.nii.gz', usedefault=True, desc='file with the mask simulated')
    out_bvec = File('bvec.sim', usedefault=True, desc='simulated b vectors')
    out_bval = File('bval.sim', usedefault=True, desc='simulated b values')
    snr = traits.Int(0, usedefault=True, desc='signal-to-noise ratio (dB)')