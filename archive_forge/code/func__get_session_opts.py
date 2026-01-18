import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def _get_session_opts(self) -> Dict[str, Any]:
    return {'headers': self.headers.copy(), 'auth': self._auth, 'timeout': self.timeout, 'verify': self.ssl_verify}