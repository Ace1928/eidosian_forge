from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
@unittest.skip('http://bugs.python.org/issue7980')
class TestThreadImport(unittest.TestCase):

    def test_strptime(self):

        def f():
            for m in range(1, 13):
                for d in range(1, 29):
                    boto.utils.parse_ts('2013-01-01T00:00:00Z')
        for _ in range(10):
            _thread.start_new_thread(f, ())
        time.sleep(3)