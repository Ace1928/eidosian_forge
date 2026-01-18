import logging
import warnings
import numpy as np
from .. import context as ctx
from ..initializer import Uniform
from .. import ndarray as nd
from .. import symbol as sym
from .base_module import BaseModule, _check_input_names
from .module import Module
from ..model import load_params
from ..name import NameManager
Creates a model from a dict mapping bucket_key to symbols and shared arg_params
        and aux_params.

        Parameters
        ----------
        sym_dict : dict mapping bucket_key to symbol
            Dict mapping bucket key to symbol
        sym_gen : function
            A function when called with a bucket key, returns a triple
            ``(symbol, data_names, label_names)``.
            provide sym_gen which was used when saving bucketing module.
        default_bucket_key : str (or any python object)
            The key for the default bucket.
        arg_params : dict
            Required for loading the BucketingModule.
            Dict of name to parameter ndarrays.
        aux_params : dict
            Required for loading the BucketingModule.
            Dict of name to auxiliary state ndarrays.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is ``cpu()``.
        work_load_list : list of number
            Default ``None``, indicating uniform workload.
        fixed_param_names: list of str
            Default ``None``, indicating no network parameters are fixed.
        state_names : list of str
            States are similar to data and label, but not provided by data iterator.
            Instead they are initialized to 0 and can be set by set_states()
        group2ctxs : dict of str to context or list of context,
                     or list of dict of str to context
            Default is `None`. Mapping the `ctx_group` attribute to the context assignment.
        compression_params : dict
            Specifies type of gradient compression and additional arguments depending
            on the type of compression being used. For example, 2bit compression requires a threshold.
            Arguments would then be {'type':'2bit', 'threshold':0.5}
            See mxnet.KVStore.set_gradient_compression method for more details on gradient compression.
        