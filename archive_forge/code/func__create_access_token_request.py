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
def _create_access_token_request(creds: Dict[str, Any], scopes: List[str]) -> Request:
    if 'private_key' in creds:
        return _create_token_request(creds['client_email'], creds['private_key'], scopes)
    elif 'refresh_token' in creds:
        return _refresh_access_token_request(refresh_token=creds['refresh_token'], client_id=creds['client_id'], client_secret=creds['client_secret'])
    else:
        raise Error('Credentials not recognized')