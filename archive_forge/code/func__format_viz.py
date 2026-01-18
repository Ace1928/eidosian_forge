import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
def _format_viz(data, viz_kind, device):
    if device is not None:
        warnings.warn('device argument is deprecated, plots now contain all device')
    buffer = pickle.dumps(data)
    buffer += b'\x00' * (3 - len(buffer) % 3)
    encoded_buffer = base64.b64encode(buffer).decode('utf-8')
    json_format = json.dumps([{'name': 'snapshot.pickle', 'base64': encoded_buffer}])
    return _memory_viz_template.replace('$VIZ_KIND', repr(viz_kind)).replace('$SNAPSHOT', json_format)