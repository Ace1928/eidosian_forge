import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
def _get_request_type(record: 'Record') -> Optional[str]:
    record_type = record.WhichOneof('record_type')
    if record_type != 'request':
        return None
    request_type = record.request.WhichOneof('request_type')
    return request_type