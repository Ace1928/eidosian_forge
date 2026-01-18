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
def _make_metadata_file(self) -> str:
    missing_text = 'This data was not captured'
    files_dir = f'{self.run_dir}/files'
    os.makedirs(files_dir, exist_ok=True)
    d = {}
    d['os'] = coalesce(self.run.os_version(), missing_text)
    d['python'] = coalesce(self.run.python_version(), missing_text)
    d['program'] = coalesce(self.run.program(), missing_text)
    d['cuda'] = coalesce(self.run.cuda_version(), missing_text)
    d['host'] = coalesce(self.run.host(), missing_text)
    d['username'] = coalesce(self.run.username(), missing_text)
    d['executable'] = coalesce(self.run.executable(), missing_text)
    gpus_used = self.run.gpus_used()
    if gpus_used is not None:
        d['gpu_devices'] = json.dumps(gpus_used)
        d['gpu_count'] = json.dumps(len(gpus_used))
    cpus_used = self.run.cpus_used()
    if cpus_used is not None:
        d['cpu_count'] = json.dumps(self.run.cpus_used())
    mem_used = self.run.memory_used()
    if mem_used is not None:
        d['memory'] = json.dumps({'total': self.run.memory_used()})
    fname = f'{files_dir}/wandb-metadata.json'
    with open(fname, 'w') as f:
        f.write(json.dumps(d))
    return fname