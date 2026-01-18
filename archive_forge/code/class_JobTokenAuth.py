from __future__ import annotations
import dataclasses
from typing import Any, BinaryIO, Dict, Optional, TYPE_CHECKING, Union
import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from requests.structures import CaseInsensitiveDict
from requests_toolbelt.multipart.encoder import MultipartEncoder  # type: ignore
from . import protocol
class JobTokenAuth(TokenAuth, AuthBase):

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers['JOB-TOKEN'] = self.token
        r.headers.pop('PRIVATE-TOKEN', None)
        r.headers.pop('Authorization', None)
        return r