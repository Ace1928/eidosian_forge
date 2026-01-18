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
class ImgSerializable(Serializable):
    """Expects a base64 string as input/output which is serialized to a filepath."""

    def serialized_info(self):
        return {'type': 'string', 'description': 'filepath on your computer (or URL) of image'}

    def api_info(self) -> dict[str, bool | dict]:
        return {'info': serializer_types['ImgSerializable'], 'serialized_info': True}

    def example_inputs(self) -> dict[str, Any]:
        return {'raw': media_data.BASE64_IMAGE, 'serialized': 'https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'}

    def serialize(self, x: str | None, load_dir: str | Path='', allow_links: bool=False) -> str | None:
        """
        Convert from human-friendly version of a file (string filepath) to a serialized
        representation (base64).
        Parameters:
            x: String path to file to serialize
            load_dir: Path to directory containing x
        """
        if not x:
            return None
        if utils.is_http_url_like(x):
            return utils.encode_url_to_base64(x)
        return utils.encode_file_to_base64(Path(load_dir) / x)

    def deserialize(self, x: str | None, save_dir: str | Path | None=None, root_url: str | None=None, hf_token: str | None=None) -> str | None:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by save_dir
        Parameters:
            x: Base64 representation of image to deserialize into a string filepath
            save_dir: Path to directory to save the deserialized image to
            root_url: Ignored
            hf_token: Ignored
        """
        if x is None or x == '':
            return None
        file = utils.decode_base64_to_file(x, dir=save_dir)
        return file.name