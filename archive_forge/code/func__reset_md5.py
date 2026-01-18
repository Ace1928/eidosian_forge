import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
def _reset_md5(self):
    self.md5sum = hashlib.md5() if self._md5 else NoopMD5()