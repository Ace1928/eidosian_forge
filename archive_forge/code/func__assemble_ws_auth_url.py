import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from time import mktime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from numpy import ndarray
@staticmethod
def _assemble_ws_auth_url(request_url: str, method: str='GET', api_key: str='', api_secret: str='') -> str:
    u = SparkLLMTextEmbeddings._parse_url(request_url)
    host = u.host
    path = u.path
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    signature_origin = 'host: {}\ndate: {}\n{} {} HTTP/1.1'.format(host, date, method, path)
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
    signature_sha_str = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (api_key, 'hmac-sha256', 'host date request-line', signature_sha_str)
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    values = {'host': host, 'date': date, 'authorization': authorization}
    return request_url + '?' + urlencode(values)