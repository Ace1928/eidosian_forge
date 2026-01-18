import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FUGUEInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='--in=%s', desc='filename of input volume')
    shift_in_file = File(exists=True, argstr='--loadshift=%s', desc='filename for reading pixel shift volume')
    phasemap_in_file = File(exists=True, argstr='--phasemap=%s', desc='filename for input phase image')
    fmap_in_file = File(exists=True, argstr='--loadfmap=%s', desc='filename for loading fieldmap (rad/s)')
    unwarped_file = File(argstr='--unwarp=%s', desc='apply unwarping and save as filename', xor=['warped_file'], requires=['in_file'])
    warped_file = File(argstr='--warp=%s', desc='apply forward warping and save as filename', xor=['unwarped_file'], requires=['in_file'])
    forward_warping = traits.Bool(False, usedefault=True, desc='apply forward warping instead of unwarping')
    dwell_to_asym_ratio = traits.Float(argstr='--dwelltoasym=%.10f', desc='set the dwell to asym time ratio')
    dwell_time = traits.Float(argstr='--dwell=%.10f', desc='set the EPI dwell time per phase-encode line - same as echo spacing - (sec)')
    asym_se_time = traits.Float(argstr='--asym=%.10f', desc='set the fieldmap asymmetric spin echo time (sec)')
    median_2dfilter = traits.Bool(argstr='--median', desc='apply 2D median filtering')
    despike_2dfilter = traits.Bool(argstr='--despike', desc='apply a 2D de-spiking filter')
    no_gap_fill = traits.Bool(argstr='--nofill', desc='do not apply gap-filling measure to the fieldmap')
    no_extend = traits.Bool(argstr='--noextend', desc='do not apply rigid-body extrapolation to the fieldmap')
    smooth2d = traits.Float(argstr='--smooth2=%.2f', desc='apply 2D Gaussian smoothing of sigma N (in mm)')
    smooth3d = traits.Float(argstr='--smooth3=%.2f', desc='apply 3D Gaussian smoothing of sigma N (in mm)')
    poly_order = traits.Int(argstr='--poly=%d', desc='apply polynomial fitting of order N')
    fourier_order = traits.Int(argstr='--fourier=%d', desc='apply Fourier (sinusoidal) fitting of order N')
    pava = traits.Bool(argstr='--pava', desc='apply monotonic enforcement via PAVA')
    despike_threshold = traits.Float(argstr='--despikethreshold=%s', desc='specify the threshold for de-spiking (default=3.0)')
    unwarp_direction = traits.Enum('x', 'y', 'z', 'x-', 'y-', 'z-', argstr='--unwarpdir=%s', desc='specifies direction of warping (default y)')
    phase_conjugate = traits.Bool(argstr='--phaseconj', desc='apply phase conjugate method of unwarping')
    icorr = traits.Bool(argstr='--icorr', requires=['shift_in_file'], desc='apply intensity correction to unwarping (pixel shift method only)')
    icorr_only = traits.Bool(argstr='--icorronly', requires=['unwarped_file'], desc='apply intensity correction only')
    mask_file = File(exists=True, argstr='--mask=%s', desc='filename for loading valid mask')
    nokspace = traits.Bool(False, argstr='--nokspace', desc='do not use k-space forward warping')
    save_shift = traits.Bool(False, xor=['save_unmasked_shift'], desc='write pixel shift volume')
    shift_out_file = File(argstr='--saveshift=%s', desc='filename for saving pixel shift volume')
    save_unmasked_shift = traits.Bool(argstr='--unmaskshift', xor=['save_shift'], desc='saves the unmasked shiftmap when using --saveshift')
    save_fmap = traits.Bool(False, xor=['save_unmasked_fmap'], desc='write field map volume')
    fmap_out_file = File(argstr='--savefmap=%s', desc='filename for saving fieldmap (rad/s)')
    save_unmasked_fmap = traits.Bool(False, argstr='--unmaskfmap', xor=['save_fmap'], desc='saves the unmasked fieldmap when using --savefmap')