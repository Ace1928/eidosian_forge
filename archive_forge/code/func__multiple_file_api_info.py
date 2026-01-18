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
def _multiple_file_api_info(self):
    return {'info': serializer_types['MultipleFileSerializable'], 'serialized_info': True}