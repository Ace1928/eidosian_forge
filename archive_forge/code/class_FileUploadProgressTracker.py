from __future__ import annotations
import asyncio
import functools
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass as python_dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import (
from urllib.parse import urlparse
import anyio
import fastapi
import gradio_client.utils as client_utils
import httpx
import multipart
from gradio_client.documentation import document
from multipart.multipart import parse_options_header
from starlette.datastructures import FormData, Headers, MutableHeaders, UploadFile
from starlette.formparsers import MultiPartException, MultipartPart
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from gradio import processing_utils, utils
from gradio.data_classes import PredictBody
from gradio.exceptions import Error
from gradio.helpers import EventData
from gradio.state_holder import SessionState
@python_dataclass
class FileUploadProgressTracker:
    deque: deque[FileUploadProgressUnit]
    is_done: bool