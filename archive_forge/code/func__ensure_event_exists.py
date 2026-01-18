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
def _ensure_event_exists(self, event: EventId) -> None:
    if event not in self.recorded_sync_states:
        logger.info('Found Event with id: %s, but no matching event creation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?', event)
        self.create_event(event)