import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class S3USEast2StorageDriver(S3StorageDriver):
    name = 'Amazon S3 (us-east-2)'
    connectionCls = S3USEast2Connection
    ex_location_name = 'us-east-2'
    region_name = 'us-east-2'