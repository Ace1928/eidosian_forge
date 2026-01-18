import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TShiftInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dTshift', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_tshift', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    tr = Str(desc='manually set the TR. You can attach suffix "s" for seconds or "ms" for milliseconds.', argstr='-TR %s')
    tzero = traits.Float(desc='align each slice to given time offset', argstr='-tzero %s', xor=['tslice'])
    tslice = traits.Int(desc='align each slice to time offset of given slice', argstr='-slice %s', xor=['tzero'])
    ignore = traits.Int(desc='ignore the first set of points specified', argstr='-ignore %s')
    interp = traits.Enum(('Fourier', 'linear', 'cubic', 'quintic', 'heptic'), desc='different interpolation methods (see 3dTshift for details) default = Fourier', argstr='-%s')
    tpattern = traits.Either(traits.Enum('alt+z', 'altplus', 'alt+z2', 'alt-z', 'altminus', 'alt-z2', 'seq+z', 'seqplus', 'seq-z', 'seqminus'), Str, desc='use specified slice time pattern rather than one in header', argstr='-tpattern %s', xor=['slice_timing'])
    slice_timing = traits.Either(File(exists=True), traits.List(traits.Float), desc='time offsets from the volume acquisition onset for each slice', argstr='-tpattern @%s', xor=['tpattern'])
    slice_encoding_direction = traits.Enum('k', 'k-', usedefault=True, desc='Direction in which slice_timing is specified (default: k). If negative,slice_timing is defined in reverse order, that is, the first entry corresponds to the slice with the largest index, and the final entry corresponds to slice index zero. Only in effect when slice_timing is passed as list, not when it is passed as file.')
    rlt = traits.Bool(desc='Before shifting, remove the mean and linear trend', argstr='-rlt')
    rltplus = traits.Bool(desc='Before shifting, remove the mean and linear trend and later put back the mean', argstr='-rlt+')