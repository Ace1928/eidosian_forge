from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd, SpaceHardware, SpaceStage
from huggingface_hub.utils import (
from packaging import version
from gradio_client import utils
from gradio_client.compatibility import EndpointV3Compatibility
from gradio_client.data_classes import ParameterInfo
from gradio_client.documentation import document
from gradio_client.exceptions import AuthenticationError
from gradio_client.utils import (
def _stream_heartbeat(self):
    while True:
        url = self.heartbeat_url.format(session_hash=self.session_hash)
        try:
            with httpx.stream('GET', url, headers=self.headers, cookies=self.cookies, verify=self.ssl_verify, timeout=20) as response:
                for _ in response.iter_lines():
                    if self._refresh_heartbeat.is_set():
                        self._refresh_heartbeat.clear()
                        break
                    if self._kill_heartbeat.is_set():
                        return
        except httpx.TransportError:
            return