from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def _convert_mask(self, states_dict, sparse_coo=True):
    """Converts the mask to sparse coo or dense depending on the `sparse_coo` argument.
        If `sparse_coo=True`, then the mask is stored as sparse coo else dense tensor
        """
    states = copy.deepcopy(states_dict)
    for state in states.values():
        if state['mask'] is not None:
            if isinstance(state['mask'], List):
                for idx in range(len(state['mask'])):
                    if sparse_coo:
                        state['mask'][idx] = state['mask'][idx].to_sparse_coo()
                    else:
                        state['mask'][idx] = state['mask'][idx].to_dense()
            elif sparse_coo:
                state['mask'] = state['mask'].to_sparse_coo()
            else:
                state['mask'] = state['mask'].to_dense()
    return states