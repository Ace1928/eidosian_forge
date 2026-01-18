import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class CreateMatrixInputSpec(TraitedSpec):
    roi_file = File(exists=True, mandatory=True, desc='Freesurfer aparc+aseg file')
    tract_file = File(exists=True, mandatory=True, desc='Trackvis tract file')
    resolution_network_file = File(exists=True, mandatory=True, desc='Parcellation files from Connectome Mapping Toolkit')
    count_region_intersections = traits.Bool(False, usedefault=True, desc='Counts all of the fiber-region traversals in the connectivity matrix (requires significantly more computational time)')
    out_matrix_file = File(genfile=True, desc='NetworkX graph describing the connectivity')
    out_matrix_mat_file = File('cmatrix.mat', usedefault=True, desc='Matlab matrix describing the connectivity')
    out_mean_fiber_length_matrix_mat_file = File(genfile=True, desc='Matlab matrix describing the mean fiber lengths between each node.')
    out_median_fiber_length_matrix_mat_file = File(genfile=True, desc='Matlab matrix describing the mean fiber lengths between each node.')
    out_fiber_length_std_matrix_mat_file = File(genfile=True, desc='Matlab matrix describing the deviation in fiber lengths connecting each node.')
    out_intersection_matrix_mat_file = File(genfile=True, desc='Matlab connectivity matrix if all region/fiber intersections are counted.')
    out_endpoint_array_name = File(genfile=True, desc='Name for the generated endpoint arrays')