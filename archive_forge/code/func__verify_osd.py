import copy
from itertools import chain
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.checkpoint.state_dict import (
def _verify_osd(self, model: nn.Module, optim: torch.optim.Optimizer, osd: Dict[str, Any], dist_osd: Dict[str, Any]) -> None:
    params = list(chain.from_iterable((g['params'] for g in optim.param_groups)))
    param_pid_mapping = dict(zip(params, range(len(params))))
    fqn_pid_mapping = {}
    for fqn, param in model.named_parameters():
        pid = param_pid_mapping[param]
        fqn_pid_mapping[fqn] = pid
        fqn_pid_mapping[pid] = fqn
    self.assertEqual(len(osd[STATE]), len(dist_osd[STATE]))
    for pid, states in osd[STATE].items():
        fqn = fqn_pid_mapping[pid]
        dist_states = dist_osd[STATE].get(fqn, None)
        self.assertIsNotNone(dist_states, fqn)
        self.assertEqual(len(states), len(dist_states))
        for key, state in states.items():
            dist_state = states.get(key, None)
            self.assertIsNotNone(dist_state)
            self._compare_tensor(state, dist_state)
    old_dist_osd_pg = dist_osd[PG]
    if len(osd[PG]) != len(dist_osd[PG]):
        self.assertTrue(len(dist_osd[PG]) > len(osd[PG]))
        new_pg = copy.deepcopy(dist_osd[PG][0])
        new_pg['params'] = []
        for dist_group in dist_osd[PG]:
            new_pg['params'].extend(dist_group['params'])
        dist_osd[PG] = [new_pg]
    self.assertEqual(len(osd[PG]), len(dist_osd[PG]))
    for group, dist_group in zip(osd[PG], dist_osd[PG]):
        self.assertEqual(len(group), len(dist_group))
        for key, value in group.items():
            dist_value = dist_group[key]
            if key == 'params':
                fqns = [fqn_pid_mapping[pid] for pid in value]
                self.assertEqual(sorted(fqns), sorted(dist_value))
            else:
                self.assertEqual(value, dist_value)
    dist_osd[PG] = old_dist_osd_pg