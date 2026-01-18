import os
from ..base import (
class TrackMergeInputSpec(CommandLineInputSpec):
    track_files = InputMultiPath(File(exists=True), desc='file containing tracks to be filtered', position=0, argstr='%s...', mandatory=True)
    output_file = File('merged_tracks.trk', desc='target file for merged tracks', position=-1, argstr='%s', usedefault=True)