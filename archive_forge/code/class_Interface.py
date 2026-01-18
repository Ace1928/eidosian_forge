import datetime
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar
import psutil
class Interface(Protocol):

    def publish_stats(self, stats: dict) -> None:
        ...

    def _publish_telemetry(self, telemetry: 'TelemetryRecord') -> None:
        ...

    def publish_files(self, files_dict: 'FilesDict') -> None:
        ...