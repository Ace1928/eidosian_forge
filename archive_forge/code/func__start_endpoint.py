from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
def _start_endpoint(self, data: dict[str, t.Any], headers: dict[str, str]) -> HttpResponse:
    tries = self.retries
    sleep = 15
    while True:
        tries -= 1
        response = self.client.put(self._uri, data=json.dumps(data), headers=headers)
        if response.status_code == 200:
            return response
        error = self._create_http_error(response)
        if response.status_code == 503:
            raise error
        if not tries:
            raise error
        display.warning(f'{error}. Trying again after {sleep} seconds.')
        time.sleep(sleep)