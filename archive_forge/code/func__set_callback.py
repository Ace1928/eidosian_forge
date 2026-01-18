import logging
from typing import Callable, Optional, Sequence, Tuple
from wandb.proto import wandb_internal_pb2 as pb
def _set_callback(self, cb: Callable[[pb.MetricRecord], None]) -> None:
    self._callback = cb