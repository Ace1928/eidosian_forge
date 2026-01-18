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
def _delete_part(conf: Config, bucket: str, name: str) -> None:
    req = Request(url=build_url('/storage/v1/b/{bucket}/o/{object}', bucket=bucket, object=name), method='DELETE', success_codes=(204, 404))
    execute_api_request(conf, req)