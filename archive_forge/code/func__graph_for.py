import contextlib
from typing import List, Tuple
import torch
def _graph_for(self, *args, **kwargs):
    return _script_method_graph_for(self, self, *args, **kwargs)