import copy
import logging
from typing import Dict, List, Optional, Tuple
import ray
import ray.cloudpickle as pickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import unflatten_dict
def _setup_zoopt(self):
    if self._metric is None and self._mode:
        self._metric = DEFAULT_METRIC
    _dim_list = []
    for k in self._dim_dict:
        self._dim_keys.append(k)
        _dim_list.append(self._dim_dict[k])
    init_samples = None
    if self._points_to_evaluate:
        logger.warning('`points_to_evaluate` is ignored by ZOOpt in versions <= 0.4.1.')
        init_samples = [Solution(x=tuple((point[dim] for dim in self._dim_keys))) for point in self._points_to_evaluate]
    dim = zoopt.Dimension2(_dim_list)
    par = zoopt.Parameter(budget=self._budget, init_samples=init_samples)
    if self._algo == 'sracos' or self._algo == 'asracos':
        from zoopt.algos.opt_algorithms.racos.sracos import SRacosTune
        self.optimizer = SRacosTune(dimension=dim, parameter=par, parallel_num=self.parallel_num, **self.kwargs)