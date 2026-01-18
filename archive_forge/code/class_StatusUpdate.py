from __future__ import annotations
import asyncio
import base64
import copy
import json
import mimetypes
import os
import pkgutil
import secrets
import shutil
import tempfile
import warnings
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, Optional, TypedDict
import fsspec.asyn
import httpx
import huggingface_hub
from huggingface_hub import SpaceStage
from websockets.legacy.protocol import WebSocketCommonProtocol
@dataclass
class StatusUpdate:
    """Update message sent from the worker thread to the Job on the main thread."""
    code: Status
    rank: int | None
    queue_size: int | None
    eta: float | None
    success: bool | None
    time: datetime | None
    progress_data: list[ProgressUnit] | None
    log: tuple[str, str] | None = None