from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
class GlobalOptimManager:
    """
    A global optimizer manager for enabling custom optimizer configs.
    """
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group['params']):
                if id(p) in self.pid2config:
                    self.index2config[group_index, p_index] = self.pid2config[id(p)]

    def override_config(self, parameters, key=None, value=None, key_value_dict=None):
        """
        Override initial optimizer config with specific hyperparameters.

        The key-values of the optimizer config for the input parameters are overridden
        This can be both, optimizer parameters like `betas` or `lr`, or it can be
        8-bit specific parameters like `optim_bits` or `percentile_clipping`.

        Arguments:
           parameters (`torch.Tensor` or `list(torch.Tensors)`):
             The input parameters.
           key (`str`):
             The hyperparamter to override.
           value:
             The hyperparameter values.
           key_value_dict (`dict`):
             A dictionary with multiple key-values to override.

        Example:

        ```py
        import torch
        import bitsandbytes as bnb

        mng = bnb.optim.GlobalOptimManager.get_instance()

        model = MyModel()
        mng.register_parameters(model.parameters()) # 1. register parameters while still on CPU

        model = model.cuda()
        # use 8-bit optimizer states for all parameters
        adam = bnb.optim.Adam(model.parameters(), lr=0.001, optim_bits=8)

        # 2. override: the parameter model.fc1.weight now uses 32-bit Adam
        mng.override_config(model.fc1.weight, 'optim_bits', 32)
        ```
        """
        self.uses_config_override = True
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}
        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:
                    self.pid2config[id(p)].update(key_value_dict)
                else:
                    self.pid2config[id(p)] = key_value_dict

    def register_module_override(self, module, param_name, config):
        self.module_weight_config_triple.append((module, param_name, config))