import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBootstrapInputSpec(TrackInputSpec):
    scheme_file = File(argstr='-schemefile %s', mandatory=True, exists=True, desc='The scheme file corresponding to the data being processed.')
    iterations = traits.Int(argstr='-iterations %d', units='NA', desc='Number of streamlines to generate at each seed point.')
    inversion = traits.Int(argstr='-inversion %s', desc='Tensor reconstruction algorithm for repetition bootstrapping.\nDefault is 1 (linear reconstruction, single tensor).')
    bsdatafiles = traits.List(File(exists=True), mandatory=True, argstr='-bsdatafile %s', desc='Specifies files containing raw data for repetition bootstrapping.\nUse -inputfile for wild bootstrap data.')
    bgmask = File(argstr='-bgmask %s', exists=True, desc="Provides the name of a file containing a background mask computed using, for example,\nFSL's bet2 program.\nThe mask file contains zero in background voxels and non-zero in foreground.")