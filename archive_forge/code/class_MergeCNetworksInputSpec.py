import os
import os.path as op
import datetime
import string
import networkx as nx
from ...utils.filemanip import split_filename
from ..base import (
from .base import CFFBaseInterface, have_cfflib
class MergeCNetworksInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='List of CFF files to extract networks from')
    out_file = File('merged_network_connectome.cff', usedefault=True, desc='Output CFF file with all the networks added')