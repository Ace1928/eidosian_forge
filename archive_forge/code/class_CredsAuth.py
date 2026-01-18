from __future__ import annotations
from lazyops.libs import lazyload
from gspread.client import Client as BaseClient
from gspread.http_client import HTTPClient as BaseHTTPClient
from typing import Optional, Union, Tuple
class CredsAuth(niquests.Auth):
    """
    An instance of this class is used to authenticate requests.
    """

    def __init__(self, auth: Credentials) -> None:
        self.auth = auth

    def __call__(self, r: 'Request'):
        if not self.auth.valid:
            self.auth.refresh(r)