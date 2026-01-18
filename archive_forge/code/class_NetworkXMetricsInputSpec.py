import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
class NetworkXMetricsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Input network')
    out_k_core = File('k_core', usedefault=True, desc='Computed k-core network stored as a NetworkX pickle.')
    out_k_shell = File('k_shell', usedefault=True, desc='Computed k-shell network stored as a NetworkX pickle.')
    out_k_crust = File('k_crust', usedefault=True, desc='Computed k-crust network stored as a NetworkX pickle.')
    treat_as_weighted_graph = traits.Bool(True, usedefault=True, desc='Some network metrics can be calculated while considering only a binarized version of the graph')
    compute_clique_related_measures = traits.Bool(False, usedefault=True, desc='Computing clique-related measures (e.g. node clique number) can be very time consuming')
    out_global_metrics_matlab = File(genfile=True, desc='Output node metrics in MATLAB .mat format')
    out_node_metrics_matlab = File(genfile=True, desc='Output node metrics in MATLAB .mat format')
    out_edge_metrics_matlab = File(genfile=True, desc='Output edge metrics in MATLAB .mat format')
    out_pickled_extra_measures = File('extra_measures', usedefault=True, desc='Network measures for group 1 that return dictionaries stored as a Pickle.')