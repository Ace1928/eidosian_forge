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
class SimpleSerializable(Serializable):
    """General class that does not perform any serialization or deserialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {'info': serializer_types['SimpleSerializable'], 'serialized_info': False}

    def example_inputs(self) -> dict[str, Any]:
        return {'raw': None, 'serialized': None}