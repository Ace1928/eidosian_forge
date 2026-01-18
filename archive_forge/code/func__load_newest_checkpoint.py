import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
def _load_newest_checkpoint(dirpath: str, ckpt_pattern: str) -> Optional[Dict]:
    """Returns the most recently modified checkpoint.

    Assumes files are saved with an ordered name, most likely by
    :obj:atomic_save.

    Args:
        dirpath: Directory in which to look for the checkpoint file.
        ckpt_pattern: File name pattern to match to find checkpoint
            files.

    Returns:
        (dict) Deserialized state dict.
    """
    import ray.cloudpickle as cloudpickle
    full_paths = glob.glob(os.path.join(dirpath, ckpt_pattern))
    if not full_paths:
        return
    most_recent_checkpoint = max(full_paths)
    with open(most_recent_checkpoint, 'rb') as f:
        checkpoint_state = cloudpickle.load(f)
    return checkpoint_state