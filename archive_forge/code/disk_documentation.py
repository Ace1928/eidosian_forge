import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from wandb.errors.term import termwarn
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
Return a new instance of the CPU metrics.