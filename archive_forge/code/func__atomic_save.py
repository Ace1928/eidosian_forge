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
def _atomic_save(state: Dict, checkpoint_dir: str, file_name: str, tmp_file_name: str):
    """Atomically saves the state object to the checkpoint directory.

    This is automatically used by Tuner().fit during a Tune job.

    Args:
        state: Object state to be serialized.
        checkpoint_dir: Directory location for the checkpoint.
        file_name: Final name of file.
        tmp_file_name: Temporary name of file.
    """
    import ray.cloudpickle as cloudpickle
    tmp_search_ckpt_path = os.path.join(checkpoint_dir, tmp_file_name)
    with open(tmp_search_ckpt_path, 'wb') as f:
        cloudpickle.dump(state, f)
    os.replace(tmp_search_ckpt_path, os.path.join(checkpoint_dir, file_name))