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