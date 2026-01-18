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
class NumberSerializable(Serializable):
    """Expects a number (int/float) as input/output but performs no serialization."""

    def api_info(self) -> dict[str, bool | dict]:
        return {'info': serializer_types['NumberSerializable'], 'serialized_info': False}

    def example_inputs(self) -> dict[str, Any]:
        return {'raw': 5, 'serialized': 5}