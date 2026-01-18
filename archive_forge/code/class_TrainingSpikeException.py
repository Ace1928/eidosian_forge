import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
class TrainingSpikeException(RuntimeError):
    """Exception to be raised with Training Spikes."""

    def __init__(self, batch_idx: int, *args: Any, **kwargs: Any):
        super().__init__(f'Training spike detected in batch {batch_idx}', *args, **kwargs)