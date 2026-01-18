import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PRELUDEInputSpec(FSLCommandInputSpec):
    complex_phase_file = File(exists=True, argstr='--complex=%s', mandatory=True, xor=['magnitude_file', 'phase_file'], desc='complex phase input volume')
    magnitude_file = File(exists=True, argstr='--abs=%s', mandatory=True, xor=['complex_phase_file'], desc='file containing magnitude image')
    phase_file = File(exists=True, argstr='--phase=%s', mandatory=True, xor=['complex_phase_file'], desc='raw phase file')
    unwrapped_phase_file = File(genfile=True, argstr='--unwrap=%s', desc='file containing unwrapepd phase', hash_files=False)
    num_partitions = traits.Int(argstr='--numphasesplit=%d', desc='number of phase partitions to use')
    labelprocess2d = traits.Bool(argstr='--labelslices', desc='does label processing in 2D (slice at a time)')
    process2d = traits.Bool(argstr='--slices', xor=['labelprocess2d'], desc='does all processing in 2D (slice at a time)')
    process3d = traits.Bool(argstr='--force3D', xor=['labelprocess2d', 'process2d'], desc='forces all processing to be full 3D')
    threshold = traits.Float(argstr='--thresh=%.10f', desc='intensity threshold for masking')
    mask_file = File(exists=True, argstr='--mask=%s', desc='filename of mask input volume')
    start = traits.Int(argstr='--start=%d', desc='first image number to process (default 0)')
    end = traits.Int(argstr='--end=%d', desc='final image number to process (default Inf)')
    savemask_file = File(argstr='--savemask=%s', desc='saving the mask volume', hash_files=False)
    rawphase_file = File(argstr='--rawphase=%s', desc='saving the raw phase output', hash_files=False)
    label_file = File(argstr='--labels=%s', desc='saving the area labels output', hash_files=False)
    removeramps = traits.Bool(argstr='--removeramps', desc='remove phase ramps during unwrapping')