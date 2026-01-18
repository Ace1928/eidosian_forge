import datetime
import logging
import queue
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple
from .assets.asset_registry import asset_registry
from .assets.interfaces import Asset, Interface
from .assets.open_metrics import OpenMetrics
from .system_info import SystemInfo
def _get_assets(self) -> List['Asset']:
    return [asset_class(interface=self.asset_interface or self.backend_interface, settings=self.settings, shutdown_event=self._shutdown_event) for asset_class in asset_registry]