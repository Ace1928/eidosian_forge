import os
import time
import logging
import warnings
from collections import namedtuple
import numpy as np
from . import io
from . import ndarray as nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from .io import DataDesc
from .base import mx_real_t
from .callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position
def _create_kvstore(kvstore, num_device, arg_params):
    """Create kvstore
    This function select and create a proper kvstore if given the kvstore type.

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore.
    num_device : int
        The number of devices
    arg_params : dict of str to `NDArray`.
        Model parameter, dict of name to `NDArray` of net's weights.
    """
    update_on_kvstore = bool(int(os.getenv('MXNET_UPDATE_ON_KVSTORE', '1')))
    if kvstore is None:
        kv = None
    elif isinstance(kvstore, kvs.KVStoreBase):
        kv = kvstore
    elif isinstance(kvstore, str):
        if num_device == 1 and 'dist' not in kvstore:
            kv = None
        else:
            kv = kvs.create(kvstore)
            if kvstore == 'local':
                max_size = max((np.prod(param.shape) for param in arg_params.values()))
                if max_size > 1024 * 1024 * 16:
                    update_on_kvstore = False
    else:
        raise TypeError('kvstore must be KVStore, str or None')
    if kv is None:
        update_on_kvstore = False
    else:
        update_on_kvstore &= kv.is_capable(kvs.KVStoreBase.OPTIMIZER)
    return (kv, update_on_kvstore)