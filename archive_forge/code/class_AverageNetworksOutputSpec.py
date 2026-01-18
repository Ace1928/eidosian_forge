import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
class AverageNetworksOutputSpec(TraitedSpec):
    gpickled_groupavg = File(desc='Average network saved as a NetworkX .pck')
    gexf_groupavg = File(desc='Average network saved as a .gexf file')
    matlab_groupavgs = OutputMultiPath(File(desc='Average network saved as a .gexf file'))