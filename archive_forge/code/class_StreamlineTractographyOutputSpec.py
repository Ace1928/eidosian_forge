import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, BaseInterfaceInputSpec, File, isdefined, traits
from .base import (
class StreamlineTractographyOutputSpec(TraitedSpec):
    tracks = File(desc='TrackVis file containing extracted streamlines')
    gfa = File(desc='The resulting GFA (generalized FA) computed using the peaks of the ODF')
    odf_peaks = File(desc='peaks computed from the odf')
    out_seeds = File(desc='file containing the (N,3) *voxel* coordinates used in seeding.')