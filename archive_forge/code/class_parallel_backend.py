from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
class parallel_backend(parallel_config):
    """Change the default backend used by Parallel inside a with block.

    .. warning::
        It is advised to use the :class:`~joblib.parallel_config` context
        manager instead, which allows more fine-grained control over the
        backend configuration.

    If ``backend`` is a string it must match a previously registered
    implementation using the :func:`~register_parallel_backend` function.

    By default the following backends are available:

    - 'loky': single-host, process-based parallelism (used by default),
    - 'threading': single-host, thread-based parallelism,
    - 'multiprocessing': legacy single-host, process-based parallelism.

    'loky' is recommended to run functions that manipulate Python objects.
    'threading' is a low-overhead alternative that is most efficient for
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or
    CPU-bound code in a few calls to native code that explicitly releases the
    GIL. Note that on some rare systems (such as Pyodide),
    multiprocessing and loky may not be available, in which case joblib
    defaults to threading.

    You can also use the `Dask <https://docs.dask.org/en/stable/>`_ joblib
    backend to distribute work across machines. This works well with
    scikit-learn estimators with the ``n_jobs`` parameter, for example::

    >>> import joblib  # doctest: +SKIP
    >>> from sklearn.model_selection import GridSearchCV  # doctest: +SKIP
    >>> from dask.distributed import Client, LocalCluster # doctest: +SKIP

    >>> # create a local Dask cluster
    >>> cluster = LocalCluster()  # doctest: +SKIP
    >>> client = Client(cluster)  # doctest: +SKIP
    >>> grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1)
    ... # doctest: +SKIP
    >>> with joblib.parallel_backend("dask", scatter=[X, y]):  # doctest: +SKIP
    ...     grid_search.fit(X, y)

    It is also possible to use the distributed 'ray' backend for distributing
    the workload to a cluster of nodes. To use the 'ray' joblib backend add
    the following lines::

     >>> from ray.util.joblib import register_ray  # doctest: +SKIP
     >>> register_ray()  # doctest: +SKIP
     >>> with parallel_backend("ray"):  # doctest: +SKIP
     ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
     [-1, -2, -3, -4, -5]

    Alternatively the backend can be passed directly as an instance.

    By default all available workers will be used (``n_jobs=-1``) unless the
    caller passes an explicit value for the ``n_jobs`` parameter.

    This is an alternative to passing a ``backend='backend_name'`` argument to
    the :class:`~Parallel` class constructor. It is particularly useful when
    calling into library code that uses joblib internally but does not expose
    the backend argument in its own API.

    >>> from operator import neg
    >>> with parallel_backend('threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    Joblib also tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 0.10

    See Also
    --------
    joblib.parallel_config : context manager to change the backend
        configuration.
    """

    def __init__(self, backend, n_jobs=-1, inner_max_num_threads=None, **backend_params):
        super().__init__(backend=backend, n_jobs=n_jobs, inner_max_num_threads=inner_max_num_threads, **backend_params)
        if self.old_parallel_config is None:
            self.old_backend_and_jobs = None
        else:
            self.old_backend_and_jobs = (self.old_parallel_config['backend'], self.old_parallel_config['n_jobs'])
        self.new_backend_and_jobs = (self.parallel_config['backend'], self.parallel_config['n_jobs'])

    def __enter__(self):
        return self.new_backend_and_jobs