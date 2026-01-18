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
class ServerMessage(str, Enum):
    send_hash = 'send_hash'
    queue_full = 'queue_full'
    estimation = 'estimation'
    send_data = 'send_data'
    process_starts = 'process_starts'
    process_generating = 'process_generating'
    process_completed = 'process_completed'
    log = 'log'
    progress = 'progress'
    heartbeat = 'heartbeat'
    server_stopped = 'server_stopped'
    unexpected_error = 'unexpected_error'
    close_stream = 'close_stream'