import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, BaseInterfaceInputSpec, File, isdefined, traits
from .base import (
class StreamlineTractographyInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input diffusion data')
    in_model = File(exists=True, desc='input f/d-ODF model extracted from.')
    tracking_mask = File(exists=True, desc='input mask within which perform tracking')
    seed_mask = File(exists=True, desc='input mask within which perform seeding')
    in_peaks = File(exists=True, desc='peaks computed from the odf')
    seed_coord = File(exists=True, desc='file containing the list of seed voxel coordinates (N,3)')
    gfa_thresh = traits.Float(0.2, mandatory=True, usedefault=True, desc='GFA threshold to compute tracking mask')
    peak_threshold = traits.Float(0.5, mandatory=True, usedefault=True, desc='threshold to consider peaks from model')
    min_angle = traits.Float(25.0, mandatory=True, usedefault=True, desc='minimum separation angle')
    multiprocess = traits.Bool(True, mandatory=True, usedefault=True, desc='use multiprocessing')
    save_seeds = traits.Bool(False, mandatory=True, usedefault=True, desc='save seeding voxels coordinates')
    num_seeds = traits.Int(10000, mandatory=True, usedefault=True, desc='desired number of tracks in tractography')
    out_prefix = traits.Str(desc='output prefix for file names')