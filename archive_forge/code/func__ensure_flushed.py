import logging
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from ..interface.interface_queue import InterfaceQueue
from ..lib import proto_util, telemetry, tracelog
from . import context, datastore, flow_control
from .settings_static import SettingsStatic
def _ensure_flushed(self, offset: int) -> None:
    if self._ds:
        self._ds.ensure_flushed(offset)