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
def _detect_config_single(func):
    """Check if func({}) works."""
    func_sig = inspect.signature(func)
    use_config_single = True
    try:
        func_sig.bind({})
    except Exception as e:
        logger.debug(str(e))
        use_config_single = False
    return use_config_single