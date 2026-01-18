import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class OverlayInputSpec(FSLCommandInputSpec):
    transparency = traits.Bool(desc='make overlay colors semi-transparent', position=1, argstr='%s', usedefault=True, default_value=True)
    out_type = traits.Enum('float', 'int', position=2, usedefault=True, argstr='%s', desc='write output with float or int')
    use_checkerboard = traits.Bool(desc='use checkerboard mask for overlay', argstr='-c', position=3)
    background_image = File(exists=True, position=4, mandatory=True, argstr='%s', desc='image to use as background')
    _xor_inputs = ('auto_thresh_bg', 'full_bg_range', 'bg_thresh')
    auto_thresh_bg = traits.Bool(desc='automatically threshold the background image', argstr='-a', position=5, xor=_xor_inputs, mandatory=True)
    full_bg_range = traits.Bool(desc='use full range of background image', argstr='-A', position=5, xor=_xor_inputs, mandatory=True)
    bg_thresh = traits.Tuple(traits.Float, traits.Float, argstr='%.3f %.3f', position=5, desc='min and max values for background intensity', xor=_xor_inputs, mandatory=True)
    stat_image = File(exists=True, position=6, mandatory=True, argstr='%s', desc='statistical image to overlay in color')
    stat_thresh = traits.Tuple(traits.Float, traits.Float, position=7, mandatory=True, argstr='%.2f %.2f', desc='min and max values for the statistical overlay')
    show_negative_stats = traits.Bool(desc='display negative statistics in overlay', xor=['stat_image2'], argstr='%s', position=8)
    stat_image2 = File(exists=True, position=9, xor=['show_negative_stats'], argstr='%s', desc='second statistical image to overlay in color')
    stat_thresh2 = traits.Tuple(traits.Float, traits.Float, position=10, desc='min and max values for second statistical overlay', argstr='%.2f %.2f')
    out_file = File(desc='combined image volume', position=-1, argstr='%s', genfile=True, hash_files=False)