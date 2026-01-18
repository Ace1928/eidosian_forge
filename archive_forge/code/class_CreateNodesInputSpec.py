import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class CreateNodesInputSpec(BaseInterfaceInputSpec):
    roi_file = File(exists=True, mandatory=True, desc='Region of interest file')
    resolution_network_file = File(exists=True, mandatory=True, desc='Parcellation file from Connectome Mapping Toolkit')
    out_filename = File('nodenetwork.pck', usedefault=True, desc='Output gpickled network with the nodes defined.')