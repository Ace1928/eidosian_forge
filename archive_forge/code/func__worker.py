import copy
import functools
import logging
import math
import os
import threading
import time
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
def _worker(shard_idx, model, sample_batch, device):
    torch.set_grad_enabled(grad_enabled)
    try:
        with NullContextManager() if device.type == 'cpu' else torch.cuda.device(device):
            loss_out = force_list(self._loss(self, model, self.dist_class, sample_batch))
            loss_out = model.custom_loss(loss_out, sample_batch)
            assert len(loss_out) == len(self._optimizers)
            grad_info = {'allreduce_latency': 0.0}
            parameters = list(model.parameters())
            all_grads = [None for _ in range(len(parameters))]
            for opt_idx, opt in enumerate(self._optimizers):
                param_indices = self.multi_gpu_param_groups[opt_idx]
                for param_idx, param in enumerate(parameters):
                    if param_idx in param_indices and param.grad is not None:
                        param.grad.data.zero_()
                loss_out[opt_idx].backward(retain_graph=True)
                grad_info.update(self.extra_grad_process(opt, loss_out[opt_idx]))
                grads = []
                for param_idx, param in enumerate(parameters):
                    if param_idx in param_indices:
                        if param.grad is not None:
                            grads.append(param.grad)
                        all_grads[param_idx] = param.grad
                if self.distributed_world_size:
                    start = time.time()
                    if torch.cuda.is_available():
                        for g in grads:
                            torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
                    else:
                        torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)
                    for param_group in opt.param_groups:
                        for p in param_group['params']:
                            if p.grad is not None:
                                p.grad /= self.distributed_world_size
                    grad_info['allreduce_latency'] += time.time() - start
        with lock:
            results[shard_idx] = (all_grads, grad_info)
    except Exception as e:
        import traceback
        with lock:
            results[shard_idx] = (ValueError(f'Error In tower {shard_idx} on device {device} during multi GPU parallel gradient calculation:: {e}\nTraceback: \n{traceback.format_exc()}\n'), e)