import abc
import copy
import inspect
import json
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import pyarrow.fs
import ray
import ray.cloudpickle as pickle
from ray._private.dict import merge_dicts
from ray.air._internal import usage as air_usage
from ray.air._internal.config import ensure_only_allowed_dataclass_keys_updated
from ray.air._internal.usage import AirEntrypoint
from ray.air.config import RunConfig, ScalingConfig
from ray.air.result import Result
from ray.train import Checkpoint
from ray.train._internal.session import _get_session
from ray.train._internal.storage import _exists_at_fs_path, get_fs_and_path
from ray.train.constants import TRAIN_DATASET_KEY
from ray.util import PublicAPI
from ray.util.annotations import DeveloperAPI

                ResourceChangingScheduler workaround.

                Ensures that the scaling config matches trial resources.

                This should be replaced with RCS returning a ScalingConfig
                in the future.
                