import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NwarpApplyInputSpec(CommandLineInputSpec):
    in_file = traits.Either(File(exists=True), traits.List(File(exists=True)), mandatory=True, argstr='-source %s', desc='the name of the dataset to be warped can be multiple datasets')
    warp = traits.String(desc='the name of the warp dataset. multiple warps can be concatenated (make sure they exist)', argstr='-nwarp %s', mandatory=True)
    inv_warp = traits.Bool(desc="After the warp specified in '-nwarp' is computed, invert it", argstr='-iwarp')
    master = File(exists=True, desc='the name of the master dataset, which defines the output grid', argstr='-master %s')
    interp = traits.Enum('wsinc5', 'NN', 'nearestneighbour', 'nearestneighbor', 'linear', 'trilinear', 'cubic', 'tricubic', 'quintic', 'triquintic', desc='defines interpolation method to use during warp', argstr='-interp %s', usedefault=True)
    ainterp = traits.Enum('NN', 'nearestneighbour', 'nearestneighbor', 'linear', 'trilinear', 'cubic', 'tricubic', 'quintic', 'triquintic', 'wsinc5', desc='specify a different interpolation method than might be used for the warp', argstr='-ainterp %s')
    out_file = File(name_template='%s_Nwarp', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    short = traits.Bool(desc='Write output dataset using 16-bit short integers, rather than the usual 32-bit floats.', argstr='-short')
    quiet = traits.Bool(desc="don't be verbose :(", argstr='-quiet', xor=['verb'])
    verb = traits.Bool(desc='be extra verbose :)', argstr='-verb', xor=['quiet'])