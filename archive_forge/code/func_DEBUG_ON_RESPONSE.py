import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
def DEBUG_ON_RESPONSE(self, statusCode: int, responseHeader: Dict[str, Union[str, int]], data: str) -> None:
    """
        Update current frame with response
        Current frame index will be attached to responseHeader
        """
    if self.DEBUG_FLAG:
        self._frameBuffer[self._frameCount][1:4] = [statusCode, responseHeader, data]
        responseHeader[self.DEBUG_HEADER_KEY] = self._frameCount