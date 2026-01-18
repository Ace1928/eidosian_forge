import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def _ensure_event_does_not_exist(self, event: EventId) -> None:
    if event in self.recorded_sync_states:
        logger.info("Found duplicate event creation in the trace for event with id: %s. Assuming the trace for event deletion wasn't caught and backfilling it now. Perhaps the sanitizer was enabled after some torch operations?", event)
        self.delete_event(event)