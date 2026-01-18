import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
class AverageNetworksInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='Networks for a group of subjects')
    resolution_network_file = File(exists=True, desc='Parcellation files from Connectome Mapping Toolkit. This is not necessary, but if included, the interface will output the statistical maps as networkx graphs.')
    group_id = traits.Str('group1', usedefault=True, desc='ID for group')
    out_gpickled_groupavg = File(desc='Average network saved as a NetworkX .pck')
    out_gexf_groupavg = File(desc='Average network saved as a .gexf file')