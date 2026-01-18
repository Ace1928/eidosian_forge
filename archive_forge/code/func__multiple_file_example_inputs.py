from __future__ import annotations
import json
import os
import secrets
import tempfile
import uuid
from pathlib import Path
from typing import Any
from gradio_client import media_data, utils
from gradio_client.data_classes import FileData
def _multiple_file_example_inputs(self) -> dict[str, Any]:
    return {'raw': [{'is_file': False, 'data': media_data.BASE64_FILE}], 'serialized': ['https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf']}