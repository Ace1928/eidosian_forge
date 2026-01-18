import datetime
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar
import psutil
def _publish_telemetry(self, telemetry: 'TelemetryRecord') -> None:
    ...