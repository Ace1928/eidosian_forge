import time
from collections import defaultdict
from functools import partial
from typing import DefaultDict
import torch
def add_fusion_group(node):
    op, name = name_for(node)
    inline_graph(node.g('Subgraph'), name + '/', node)