import os.path as op
import numpy as np
import networkx as nx
import pickle
from ... import logging
from ..base import (
from .base import have_cv
class NetworkBasedStatisticOutputSpec(TraitedSpec):
    nbs_network = File(exists=True, desc='Output network with edges identified by the NBS')
    nbs_pval_network = File(exists=True, desc='Output network with p-values to weight the edges identified by the NBS')
    network_files = OutputMultiPath(File(exists=True), desc='Output network with edges identified by the NBS')