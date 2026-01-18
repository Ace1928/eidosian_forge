import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import (
from .api import (
from .utils import _delay, _PeriodicTimer
def _remove_participant_epilogue(state: _RendezvousState, settings: RendezvousSettings) -> None:
    if state.complete:
        if not state.participants:
            state.complete = False
            state.round += 1
    elif len(state.participants) < settings.min_nodes:
        state.deadline = None