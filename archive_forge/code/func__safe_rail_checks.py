from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
@staticmethod
def _safe_rail_checks(args):
    """Makes sure that some of the functions and attributes are not passed incorrectly
        """
    features, feature_dim = (args['features'], args['feature_dim'])
    if features is not None:
        assert feature_dim is not None, 'need feature dim to select features'
    fn_keys = ['aggregate_fn', 'reduce_fn', 'mask_fn']
    for key in fn_keys:
        fn = args[key]
        assert callable(fn), 'function should be callable'