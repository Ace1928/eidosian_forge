import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import urllib3
from blobfile import _common as common
from blobfile._common import (
def _refresh_access_token_request(client_id: str, client_secret: str, refresh_token: str) -> Request:
    data = {'grant_type': 'refresh_token', 'refresh_token': refresh_token, 'client_id': client_id, 'client_secret': client_secret}
    return Request(url='https://www.googleapis.com/oauth2/v4/token', method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded'}, data=urllib.parse.urlencode(data).encode('utf8'))