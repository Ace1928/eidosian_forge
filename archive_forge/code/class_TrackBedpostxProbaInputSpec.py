import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBedpostxProbaInputSpec(TrackInputSpec):
    bedpostxdir = Directory(argstr='-bedpostxdir %s', mandatory=True, exists=True, desc='Directory containing bedpostx output')
    min_vol_frac = traits.Float(argstr='-bedpostxminf %d', units='NA', desc='Zeros out compartments in bedpostx data with a mean volume fraction f of less than min_vol_frac.  The default is 0.01.')
    iterations = traits.Int(argstr='-iterations %d', units='NA', desc='Number of streamlines to generate at each seed point. The default is 1.')