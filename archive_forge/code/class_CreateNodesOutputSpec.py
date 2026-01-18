import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class CreateNodesOutputSpec(TraitedSpec):
    node_network = File(desc='Output gpickled network with the nodes defined.')