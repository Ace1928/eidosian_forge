import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SlicerInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, position=1, argstr='%s', mandatory=True, desc='input volume')
    image_edges = File(exists=True, position=2, argstr='%s', desc='volume to display edge overlay for (useful for checking registration')
    label_slices = traits.Bool(position=3, argstr='-L', desc='display slice number', usedefault=True, default_value=True)
    colour_map = File(exists=True, position=4, argstr='-l %s', desc='use different colour map from that stored in nifti header')
    intensity_range = traits.Tuple(traits.Float, traits.Float, position=5, argstr='-i %.3f %.3f', desc='min and max intensities to display')
    threshold_edges = traits.Float(position=6, argstr='-e %.3f', desc='use threshold for edges')
    dither_edges = traits.Bool(position=7, argstr='-t', desc='produce semi-transparent (dithered) edges')
    nearest_neighbour = traits.Bool(position=8, argstr='-n', desc='use nearest neighbor interpolation for output')
    show_orientation = traits.Bool(position=9, argstr='%s', usedefault=True, default_value=True, desc='label left-right orientation')
    _xor_options = ('single_slice', 'middle_slices', 'all_axial', 'sample_axial')
    single_slice = traits.Enum('x', 'y', 'z', position=10, argstr='-%s', xor=_xor_options, requires=['slice_number'], desc='output picture of single slice in the x, y, or z plane')
    slice_number = traits.Int(position=11, argstr='-%d', desc='slice number to save in picture')
    middle_slices = traits.Bool(position=10, argstr='-a', xor=_xor_options, desc='output picture of mid-sagittal, axial, and coronal slices')
    all_axial = traits.Bool(position=10, argstr='-A', xor=_xor_options, requires=['image_width'], desc='output all axial slices into one picture')
    sample_axial = traits.Int(position=10, argstr='-S %d', xor=_xor_options, requires=['image_width'], desc='output every n axial slices into one picture')
    image_width = traits.Int(position=-2, argstr='%d', desc='max picture width')
    out_file = File(position=-1, genfile=True, argstr='%s', desc='picture to write', hash_files=False)
    scaling = traits.Float(position=0, argstr='-s %f', desc='image scale')