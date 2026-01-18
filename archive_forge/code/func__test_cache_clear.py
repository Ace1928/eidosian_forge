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
def _test_cache_clear(self, target='both', supported=True, forbidden=False):
    args = self._make_args({'target': target})
    with mock.patch.object(self.gc.cache, 'clear') as mocked_cache_clear:
        if supported:
            mocked_cache_clear.return_value = None
        else:
            mocked_cache_clear.side_effect = exc.HTTPNotImplemented
        if forbidden:
            mocked_cache_clear.side_effect = exc.HTTPForbidden
        test_shell.do_cache_clear(self.gc, args)
        if supported:
            mocked_cache_clear.mocked_cache_clear(target)