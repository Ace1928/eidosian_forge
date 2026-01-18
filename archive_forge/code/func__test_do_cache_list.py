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
def _test_do_cache_list(self, supported=True):
    args = self._make_args({})
    expected_output = {'cached_images': [{'image_id': 'pass', 'last_accessed': 0, 'last_modified': 0, 'size': 'fake_size', 'hits': 'fake_hits'}], 'queued_images': ['fake_image']}
    with mock.patch.object(self.gc.cache, 'list') as mocked_cache_list:
        if supported:
            mocked_cache_list.return_value = expected_output
        else:
            mocked_cache_list.side_effect = exc.HTTPNotImplemented
        test_shell.do_cache_list(self.gc, args)
        mocked_cache_list.assert_called()
        if supported:
            utils.print_cached_images.assert_called_once_with(expected_output)