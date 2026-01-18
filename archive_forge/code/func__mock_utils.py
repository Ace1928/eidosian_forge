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
def _mock_utils(self):
    utils.print_list = mock.Mock()
    utils.print_dict = mock.Mock()
    utils.save_image = mock.Mock()
    utils.print_dict_list = mock.Mock()
    utils.print_cached_images = mock.Mock()