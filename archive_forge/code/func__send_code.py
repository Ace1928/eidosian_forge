import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
def _send_code():
    target_url = self.conn._redirect_uri_with_port
    params = {'state': state, 'code': expected_code}
    params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    requests.get(url=target_url, params=params)