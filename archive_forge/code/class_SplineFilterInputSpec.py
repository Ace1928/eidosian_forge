import os
from ..base import (
class SplineFilterInputSpec(CommandLineInputSpec):
    track_file = File(exists=True, desc='file containing tracks to be filtered', position=0, argstr='%s', mandatory=True)
    step_length = traits.Float(desc='in the unit of minimum voxel size', position=1, argstr='%f', mandatory=True)
    output_file = File('spline_tracks.trk', desc='target file for smoothed tracks', position=2, argstr='%s', usedefault=True)