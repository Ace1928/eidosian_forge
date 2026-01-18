import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class FWHMxInputSpec(CommandLineInputSpec):
    in_file = File(desc='input dataset', argstr='-input %s', mandatory=True, exists=True)
    out_file = File(argstr='> %s', name_source='in_file', name_template='%s_fwhmx.out', position=-1, keep_extension=False, desc='output file')
    out_subbricks = File(argstr='-out %s', name_source='in_file', name_template='%s_subbricks.out', keep_extension=False, desc='output file listing the subbricks FWHM')
    mask = File(desc='use only voxels that are nonzero in mask', argstr='-mask %s', exists=True)
    automask = traits.Bool(False, usedefault=True, argstr='-automask', desc='compute a mask from THIS dataset, a la 3dAutomask')
    detrend = traits.Either(traits.Bool(), traits.Int(), default=False, argstr='-detrend', xor=['demed'], usedefault=True, desc='instead of demed (0th order detrending), detrend to the specified order.  If order is not given, the program picks q=NT/30. -detrend disables -demed, and includes -unif.')
    demed = traits.Bool(False, argstr='-demed', xor=['detrend'], desc="If the input dataset has more than one sub-brick (e.g., has a time axis), then subtract the median of each voxel's time series before processing FWHM. This will tend to remove intrinsic spatial structure and leave behind the noise.")
    unif = traits.Bool(False, argstr='-unif', desc="If the input dataset has more than one sub-brick, then normalize each voxel's time series to have the same MAD before processing FWHM.")
    out_detrend = File(argstr='-detprefix %s', name_source='in_file', name_template='%s_detrend', keep_extension=False, desc='Save the detrended file into a dataset')
    geom = traits.Bool(argstr='-geom', xor=['arith'], desc='if in_file has more than one sub-brick, compute the final estimate as the geometric mean of the individual sub-brick FWHM estimates')
    arith = traits.Bool(argstr='-arith', xor=['geom'], desc='if in_file has more than one sub-brick, compute the final estimate as the arithmetic mean of the individual sub-brick FWHM estimates')
    combine = traits.Bool(argstr='-combine', desc='combine the final measurements along each axis')
    compat = traits.Bool(argstr='-compat', desc='be compatible with the older 3dFWHM')
    acf = traits.Either(traits.Bool(), File(), traits.Tuple(File(exists=True), traits.Float()), default=False, usedefault=True, argstr='-acf', desc='computes the spatial autocorrelation')