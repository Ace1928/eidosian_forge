import threading
from abc import abstractmethod
from typing import Optional
from wandb.proto import wandb_internal_pb2 as pb
def _set_object(self, obj: pb.Result) -> None:
    self._object = obj
    self._object_ready.set()