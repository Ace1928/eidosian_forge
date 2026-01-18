from __future__ import annotations
import base64
import hashlib
import hmac
import json
import logging
import queue
import threading
from datetime import datetime
from queue import Queue
from time import mktime
from typing import Any, Dict, Generator, Iterator, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
@staticmethod
def _create_url(api_url: str, api_key: str, api_secret: str) -> str:
    """
        Generate a request url with an api key and an api secret.
        """
    date = format_date_time(mktime(datetime.now().timetuple()))
    parsed_url = urlparse(api_url)
    host = parsed_url.netloc
    path = parsed_url.path
    signature_origin = f'host: {host}\ndate: {date}\nGET {path} HTTP/1.1'
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256",         headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    params_dict = {'authorization': authorization, 'date': date, 'host': host}
    encoded_params = urlencode(params_dict)
    url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, encoded_params, parsed_url.fragment))
    return url