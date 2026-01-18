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
class FileSerializable(Serializable):
    """Expects a dict with base64 representation of object as input/output which is serialized to a filepath."""

    def __init__(self) -> None:
        self.stream = None
        self.stream_name = None
        super().__init__()

    def serialized_info(self):
        return self._single_file_serialized_info()

    def _single_file_api_info(self):
        return {'info': serializer_types['SingleFileSerializable'], 'serialized_info': True}

    def _single_file_serialized_info(self):
        return {'type': 'string', 'description': 'filepath on your computer (or URL) of file'}

    def _multiple_file_serialized_info(self):
        return {'type': 'array', 'description': 'List of filepath(s) or URL(s) to files', 'items': {'type': 'string', 'description': 'filepath on your computer (or URL) of file'}}

    def _multiple_file_api_info(self):
        return {'info': serializer_types['MultipleFileSerializable'], 'serialized_info': True}

    def api_info(self) -> dict[str, dict | bool]:
        return self._single_file_api_info()

    def example_inputs(self) -> dict[str, Any]:
        return self._single_file_example_inputs()

    def _single_file_example_inputs(self) -> dict[str, Any]:
        return {'raw': {'is_file': False, 'data': media_data.BASE64_FILE}, 'serialized': 'https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf'}

    def _multiple_file_example_inputs(self) -> dict[str, Any]:
        return {'raw': [{'is_file': False, 'data': media_data.BASE64_FILE}], 'serialized': ['https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf']}

    def _serialize_single(self, x: str | FileData | None, load_dir: str | Path='', allow_links: bool=False) -> FileData | None:
        if x is None or isinstance(x, dict):
            return x
        if utils.is_http_url_like(x):
            filename = x
            size = None
        else:
            filename = str(Path(load_dir) / x)
            size = Path(filename).stat().st_size
        return {'name': filename or None, 'data': None if allow_links else utils.encode_url_or_file_to_base64(filename), 'orig_name': Path(filename).name, 'size': size}

    def _setup_stream(self, url, hf_token):
        return utils.download_byte_stream(url, hf_token)

    def _deserialize_single(self, x: str | FileData | None, save_dir: str | None=None, root_url: str | None=None, hf_token: str | None=None) -> str | None:
        if x is None:
            return None
        if isinstance(x, str):
            file_name = utils.decode_base64_to_file(x, dir=save_dir).name
        elif isinstance(x, dict):
            if x.get('is_file'):
                filepath = x.get('name')
                if filepath is None:
                    raise ValueError(f"The 'name' field is missing in {x}")
                if root_url is not None:
                    file_name = utils.download_tmp_copy_of_file(root_url + 'file=' + filepath, hf_token=hf_token, dir=save_dir)
                else:
                    file_name = utils.create_tmp_copy_of_file(filepath, dir=save_dir)
            elif x.get('is_stream'):
                if not (x['name'] and root_url and save_dir):
                    raise ValueError('name and root_url and save_dir must all be present')
                if not self.stream or self.stream_name != x['name']:
                    self.stream = self._setup_stream(root_url + 'stream/' + x['name'], hf_token=hf_token)
                    self.stream_name = x['name']
                chunk = next(self.stream)
                path = Path(save_dir or tempfile.gettempdir()) / secrets.token_hex(20)
                path.mkdir(parents=True, exist_ok=True)
                path = path / x.get('orig_name', 'output')
                path.write_bytes(chunk)
                file_name = str(path)
            else:
                data = x.get('data')
                if data is None:
                    raise ValueError(f"The 'data' field is missing in {x}")
                file_name = utils.decode_base64_to_file(data, dir=save_dir).name
        else:
            raise ValueError(f'A FileSerializable component can only deserialize a string or a dict, not a {type(x)}: {x}')
        return file_name

    def serialize(self, x: str | FileData | None | list[str | FileData | None], load_dir: str | Path='', allow_links: bool=False) -> FileData | None | list[FileData | None]:
        """
        Convert from human-friendly version of a file (string filepath) to a
        serialized representation (base64)
        Parameters:
            x: String path to file to serialize
            load_dir: Path to directory containing x
            allow_links: Will allow path returns instead of raw file content
        """
        if x is None or x == '':
            return None
        if isinstance(x, list):
            return [self._serialize_single(f, load_dir, allow_links) for f in x]
        else:
            return self._serialize_single(x, load_dir, allow_links)

    def deserialize(self, x: str | FileData | None | list[str | FileData | None], save_dir: Path | str | None=None, root_url: str | None=None, hf_token: str | None=None) -> str | None | list[str | None]:
        """
        Convert from serialized representation of a file (base64) to a human-friendly
        version (string filepath). Optionally, save the file to the directory specified by `save_dir`
        Parameters:
            x: Base64 representation of file to deserialize into a string filepath
            save_dir: Path to directory to save the deserialized file to
            root_url: If this component is loaded from an external Space, this is the URL of the Space.
            hf_token: If this component is loaded from an external private Space, this is the access token for the Space
        """
        if x is None:
            return None
        if isinstance(save_dir, Path):
            save_dir = str(save_dir)
        if isinstance(x, list):
            return [self._deserialize_single(f, save_dir=save_dir, root_url=root_url, hf_token=hf_token) for f in x]
        else:
            return self._deserialize_single(x, save_dir=save_dir, root_url=root_url, hf_token=hf_token)