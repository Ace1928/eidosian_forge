import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class VolregInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dvolreg', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    in_weight_volume = traits.Either(traits.Tuple(File(exists=True), traits.Int), File(exists=True), desc='weights for each voxel specified by a file with an optional volume number (defaults to 0)', argstr="-weight '%s[%d]'")
    out_file = File(name_template='%s_volreg', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    basefile = File(desc='base file for registration', argstr='-base %s', position=-6, exists=True)
    zpad = traits.Int(desc="Zeropad around the edges by 'n' voxels during rotations", argstr='-zpad %d', position=-5)
    md1d_file = File(name_template='%s_md.1D', desc='max displacement output file', argstr='-maxdisp1D %s', name_source='in_file', keep_extension=True, position=-4)
    oned_file = File(name_template='%s.1D', desc='1D movement parameters output file', argstr='-1Dfile %s', name_source='in_file', keep_extension=True)
    verbose = traits.Bool(desc='more detailed description of the process', argstr='-verbose')
    timeshift = traits.Bool(desc='time shift to mean slice time offset', argstr='-tshift 0')
    copyorigin = traits.Bool(desc='copy base file origin coords to output', argstr='-twodup')
    oned_matrix_save = File(name_template='%s.aff12.1D', desc='Save the matrix transformation', argstr='-1Dmatrix_save %s', keep_extension=True, name_source='in_file')
    interp = traits.Enum(('Fourier', 'cubic', 'heptic', 'quintic', 'linear'), desc='spatial interpolation methods [default = heptic]', argstr='-%s')