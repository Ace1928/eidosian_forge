import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def _mock_glance_client(self):
    my_mocked_gc = mock.Mock()
    my_mocked_gc.schemas.return_value = 'test'
    my_mocked_gc.get.return_value = {}
    return my_mocked_gc