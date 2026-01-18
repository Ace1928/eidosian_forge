import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def _set_request_ids(self, resp):
    if self._self_request_ids == []:
        req_id = _extract_request_id(resp)
        self._self_request_ids = [req_id]