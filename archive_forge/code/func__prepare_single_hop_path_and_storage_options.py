import glob
import io
import os
import posixpath
import re
import tarfile
import time
import xml.dom.minidom
import zipfile
from asyncio import TimeoutError
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import fsspec
from aiohttp.client_exceptions import ClientError
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from .. import config
from ..filesystems import COMPRESSION_FILESYSTEMS
from ..utils.file_utils import (
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
def _prepare_single_hop_path_and_storage_options(urlpath: str, download_config: Optional[DownloadConfig]=None) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Prepare the URL and the kwargs that must be passed to the HttpFileSystem or to requests.get/head

    In particular it resolves google drive URLs
    It also adds the authentication headers for the Hugging Face Hub, for both https:// and hf:// paths.

    Storage options are formatted in the form {protocol: storage_options_for_protocol}
    """
    token = None if download_config is None else download_config.token
    if urlpath.startswith(config.HF_ENDPOINT) and '/resolve/' in urlpath:
        urlpath = 'hf://' + urlpath[len(config.HF_ENDPOINT) + 1:].replace('/resolve/', '@', 1)
    protocol = urlpath.split('://')[0] if '://' in urlpath else 'file'
    if download_config is not None and protocol in download_config.storage_options:
        storage_options = download_config.storage_options[protocol]
    elif download_config is not None and protocol not in download_config.storage_options:
        storage_options = {option_name: option_value for option_name, option_value in download_config.storage_options.items() if option_name not in fsspec.available_protocols()}
    else:
        storage_options = {}
    if storage_options:
        storage_options = {protocol: storage_options}
    if protocol in ['http', 'https']:
        storage_options[protocol] = {'headers': {**get_authentication_headers_for_url(urlpath, token=token), 'user-agent': get_datasets_user_agent()}, 'client_kwargs': {'trust_env': True}, **storage_options.get(protocol, {})}
        if 'drive.google.com' in urlpath:
            response = http_head(urlpath)
            cookies = None
            for k, v in response.cookies.items():
                if k.startswith('download_warning'):
                    urlpath += '&confirm=' + v
                    cookies = response.cookies
                    storage_options[protocol] = {'cookies': cookies, **storage_options.get(protocol, {})}
        if 'drive.google.com' in urlpath and 'confirm=' not in urlpath:
            urlpath += '&confirm=t'
        if urlpath.startswith('https://raw.githubusercontent.com/'):
            storage_options[protocol]['headers']['Accept-Encoding'] = 'identity'
    elif protocol == 'hf':
        storage_options[protocol] = {'token': token, 'endpoint': config.HF_ENDPOINT, **storage_options.get(protocol, {})}
        if config.HF_HUB_VERSION < version.parse('0.21.0'):
            storage_options[protocol]['block_size'] = 'default'
    return (urlpath, storage_options)