import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
Flow Control.

States:
    FORWARDING
    PAUSING

New messages:
    pb.SenderMarkRequest    writer -> sender (empty message)
    pb.StatusReportRequest  sender -> writer (reports current sender progress)
    pb.SenderReadRequest    writer -> sender (requests read of transaction log)

Thresholds:
    Threshold_High_MaxOutstandingData      - When above this, stop sending requests to sender
    Threshold_Mid_StartSendingReadRequests - When below this, start sending read requests
    Threshold_Low_RestartSendingData       - When below this, start sending normal records

State machine:
    FORWARDING
      -> PAUSED if should_pause
         There is too much work outstanding to the sender thread, after the current request
         lets stop sending data.
    PAUSING
      -> FORWARDING if should_unpause
      -> PAUSING if should_recover
      -> PAUSING if should_quiesce

