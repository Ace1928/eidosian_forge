from collections import deque
from contextlib import contextmanager
import threading
from typing import (
import torch
from torch import Tensor
import torch.autograd
from .dependency import fork, join
from .microbatch import Batch
from .phony import get_phony
class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: Function, batch: Batch) -> None:
        self.function = function
        self.batch = batch
        self.recomputed: Deque[Recomputed] = deque(maxlen=1)
        self.rng_states: Deque[RNGStates] = deque(maxlen=1)

    def checkpoint(self) -> Batch:
        """Return a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        inputs = tuple(self.batch)
        phony = get_phony(self.batch.get_device(), requires_grad=True)
        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *inputs)
        if isinstance(output, tuple):
            output = tuple([x.detach() if torch.is_tensor(x) and (not x.is_floating_point()) else x for x in output])
        return Batch(output)

    def recompute(self, batch: Batch) -> None:
        """Apply :class:`Recompute` to the batch in place."""
        input_atomic = self.batch.atomic
        inputs = tuple(self.batch)
        tensor_idx = batch.find_tensor_idx()
        batch[tensor_idx], phony = fork(batch[tensor_idx])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *inputs)
        batch[tensor_idx] = join(batch[tensor_idx], phony)