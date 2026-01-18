import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class FilterTracksInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input tracks to be filtered')
    include_xor = ['include_file', 'include_spec']
    include_file = File(exists=True, argstr='-include %s', desc='inclusion file', xor=include_xor)
    include_spec = traits.List(traits.Float, desc='inclusion specification in mm and radius (x y z r)', position=2, argstr='-include %s', minlen=4, maxlen=4, sep=',', units='mm', xor=include_xor)
    exclude_xor = ['exclude_file', 'exclude_spec']
    exclude_file = File(exists=True, argstr='-exclude %s', desc='exclusion file', xor=exclude_xor)
    exclude_spec = traits.List(traits.Float, desc='exclusion specification in mm and radius (x y z r)', position=2, argstr='-exclude %s', minlen=4, maxlen=4, sep=',', units='mm', xor=exclude_xor)
    minimum_tract_length = traits.Float(argstr='-minlength %s', units='mm', desc='Sets the minimum length of any track in millimeters (default is 10 mm).')
    out_file = File(argstr='%s', position=-1, desc='Output filtered track filename', name_source=['in_file'], hash_files=False, name_template='%s_filt')
    no_mask_interpolation = traits.Bool(argstr='-nomaskinterp', desc='Turns off trilinear interpolation of mask images.')
    invert = traits.Bool(argstr='-invert', desc='invert the matching process, so that tracks that wouldotherwise have been included are now excluded and vice-versa.')
    quiet = traits.Bool(argstr='-quiet', position=1, desc='Do not display information messages or progress status.')
    debug = traits.Bool(argstr='-debug', position=1, desc='Display debugging messages.')