import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, BaseInterfaceInputSpec, File, isdefined, traits
from .base import (
class TrackDensityMapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='The input TrackVis track file')
    reference = File(exists=True, desc='A reference file to define RAS coordinates space')
    points_space = traits.Enum('rasmm', 'voxel', None, usedefault=True, desc='coordinates of trk file')
    voxel_dims = traits.List(traits.Float, minlen=3, maxlen=3, desc='The size of each voxel in mm.')
    data_dims = traits.List(traits.Int, minlen=3, maxlen=3, desc='The size of the image in voxels.')
    out_filename = File('tdi.nii', usedefault=True, desc='The output filename for the tracks in TrackVis (.trk) format')