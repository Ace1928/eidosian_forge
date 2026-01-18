import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk import wandb_sweep
from wandb.sdk.launch.sweeps.utils import (
from wandb.util import get_module
def _get_runs_status(metrics):
    categories = [name for name, _ in sweeps.RunState.__members__.items()] + ['unknown']
    mlist = []
    for c in categories:
        if not metrics.get(c):
            continue
        mlist.append('%s: %d' % (c.capitalize(), metrics[c]))
    s = ', '.join(mlist)
    return s