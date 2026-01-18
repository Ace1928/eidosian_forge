from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
import aiofiles
import httpx
import numpy as np
from gradio_client import utils as client_utils
from PIL import Image, ImageOps, PngImagePlugin
from gradio import utils, wasm_utils
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.utils import abspath, get_upload_folder, is_in_or_equal
class PyodideHttpTransport(httpx.AsyncBaseTransport):

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        method = request.method
        headers = dict(request.headers)
        body = None if method in ['GET', 'HEAD'] else await request.aread()
        response = await pyodide.http.pyfetch(url, method=method, headers=headers, body=body)
        return httpx.Response(status_code=response.status, headers=response.headers, stream=PyodideHttpResponseAsyncByteStream(response))