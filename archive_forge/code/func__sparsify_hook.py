from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def _sparsify_hook(self, name):
    """Returns hook that applies sparsification mask to input entering the attached layer
        """
    mask = self.get_mask(name)
    features = self.data_groups[name]['features']
    feature_dim = self.data_groups[name]['feature_dim']

    def hook(module, input):
        input_data = input[0]
        if features is None:
            return input_data * mask
        else:
            for feature_idx in range(0, len(features)):
                feature = torch.Tensor([features[feature_idx]]).long().to(input_data.device)
                sparsified = torch.index_select(input_data, feature_dim, feature) * mask[feature_idx]
                input_data.index_copy_(feature_dim, feature, sparsified)
            return input_data
    return hook