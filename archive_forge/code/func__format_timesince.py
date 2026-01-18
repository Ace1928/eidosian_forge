import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
def _format_timesince(ts: float) -> str:
    """Format timestamp in seconds into a human-readable string, relative to now.

    Vaguely inspired by Django's `timesince` formatter.
    """
    delta = time.time() - ts
    if delta < 20:
        return 'a few seconds ago'
    for label, divider, max_value in _TIMESINCE_CHUNKS:
        value = round(delta / divider)
        if max_value is not None and value <= max_value:
            break
    return f'{value} {label}{('s' if value > 1 else '')} ago'