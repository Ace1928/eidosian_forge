import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
@dataclasses.dataclass
class RunAttrs:
    """Simple data class for run attributes."""
    name: str
    display_name: str
    description: str
    sweep_name: str
    project: Dict[str, Any]
    config: Dict[str, Any]
    remote: Optional[str] = None
    commit: Optional[str] = None