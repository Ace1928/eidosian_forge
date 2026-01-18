from abc import abstractmethod
from typing import TYPE_CHECKING, Optional
from wandb.proto import wandb_server_pb2 as spb
@abstractmethod
def _svc_inform_start(self, settings: 'wandb_settings_pb2.Settings', run_id: str) -> None:
    raise NotImplementedError