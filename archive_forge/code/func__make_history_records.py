import json
import logging
import math
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
from google.protobuf.json_format import ParseDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from wandb import Artifact
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_settings_pb2
from wandb.proto import wandb_telemetry_pb2 as telem_pb
from wandb.sdk.interface.interface import file_policy_to_enum
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context
from wandb.sdk.internal.sender import SendManager
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.util import coalesce, recursive_cast_dictlike_to_dict
from .protocols import ImporterRun
def _make_history_records(self) -> Iterable[pb.Record]:
    for metrics in self.run.metrics():
        history = pb.HistoryRecord()
        for k, v in metrics.items():
            item = history.item.add()
            item.key = k
            if isinstance(v, float) and math.isnan(v) or v == 'NaN':
                v = np.NaN
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            else:
                v = json.dumps(v)
            item.value_json = v
        rec = self.interface._make_record(history=history)
        yield rec