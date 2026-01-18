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
def _make_run_record(self) -> pb.Record:
    run = pb.RunRecord()
    run.run_id = self.run.run_id()
    run.entity = self.run.entity()
    run.project = self.run.project()
    run.display_name = coalesce(self.run.display_name())
    run.notes = coalesce(self.run.notes(), '')
    run.tags.extend(coalesce(self.run.tags(), []))
    run.start_time.FromMilliseconds(self.run.start_time())
    host = self.run.host()
    if host is not None:
        run.host = host
    runtime = self.run.runtime()
    if runtime is not None:
        run.runtime = runtime
    run_group = self.run.run_group()
    if run_group is not None:
        run.run_group = run_group
    config = self.run.config()
    if '_wandb' not in config:
        config['_wandb'] = {}
    config['_wandb']['code_path'] = self.run.code_path()
    config['_wandb']['python_version'] = self.run.python_version()
    config['_wandb']['cli_version'] = self.run.cli_version()
    self.interface._make_config(data=config, obj=run.config)
    return self.interface._make_record(run=run)