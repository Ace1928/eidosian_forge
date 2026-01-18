import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
class NetworkXMetricsOutputSpec(TraitedSpec):
    gpickled_network_files = OutputMultiPath(File(desc='Output gpickled network files'))
    matlab_matrix_files = OutputMultiPath(File(desc='Output network metrics in MATLAB .mat format'))
    global_measures_matlab = File(desc='Output global metrics in MATLAB .mat format')
    node_measures_matlab = File(desc='Output node metrics in MATLAB .mat format')
    edge_measures_matlab = File(desc='Output edge metrics in MATLAB .mat format')
    node_measure_networks = OutputMultiPath(File(desc='Output gpickled network files for all node-based measures'))
    edge_measure_networks = OutputMultiPath(File(desc='Output gpickled network files for all edge-based measures'))
    k_networks = OutputMultiPath(File(desc='Output gpickled network files for the k-core, k-shell, and k-crust networks'))
    k_core = File(desc='Computed k-core network stored as a NetworkX pickle.')
    k_shell = File(desc='Computed k-shell network stored as a NetworkX pickle.')
    k_crust = File(desc='Computed k-crust network stored as a NetworkX pickle.')
    pickled_extra_measures = File(desc='Network measures for the group that return dictionaries, stored as a Pickle.')
    matlab_dict_measures = OutputMultiPath(File(desc='Network measures for the group that return dictionaries, stored as matlab matrices.'))