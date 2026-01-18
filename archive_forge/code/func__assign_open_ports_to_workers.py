import operator
import socket
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse
import numpy as np
import scipy.sparse as ss
from .basic import LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning
from .compat import (DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED, Client, Future, LGBMNotFittedError, concat,
from .sklearn import (LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor, _LGBM_ScikitCustomObjectiveFunction,
def _assign_open_ports_to_workers(client: Client, workers: List[str]) -> Tuple[Dict[str, Future], Dict[str, int]]:
    """Assign an open port to each worker.

    Returns
    -------
    worker_to_socket_future: dict
        mapping from worker address to a future pointing to the remote socket.
    worker_to_port: dict
        mapping from worker address to an open port in the worker's host.
    """
    worker_to_future = {}
    for worker in workers:
        worker_to_future[worker] = client.submit(_acquire_port, workers=[worker], allow_other_workers=False, pure=False)
    worker_to_socket_future = {}
    worker_to_port_future = {}
    for worker, socket_future in worker_to_future.items():
        worker_to_socket_future[worker] = client.submit(operator.itemgetter(0), socket_future)
        worker_to_port_future[worker] = client.submit(operator.itemgetter(1), socket_future)
    worker_to_port = client.gather(worker_to_port_future)
    return (worker_to_socket_future, worker_to_port)