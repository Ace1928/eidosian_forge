from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
@contextmanager
def _mock_boto_config(boto_config_dict):
    """"Mock boto config replacing any exiting config.

  The util.SetBotoConfigForTest has a use_existing_config flag that can be
  set to False, but it does not work if the config has been already loaded,
  which is the case for all unit tests that do not use RunCommand method.

  Args:
    boto_config_dict. A dict with key=<boto section name> and value=<a dict
      of boto field name and the value for that field>.
  """

    def _config_get_side_effect(section, key, default_value=None):
        return boto_config_dict.get(section, {}).get(key, default_value)
    with mock.patch.object(config, 'get', autospec=True) as mock_get:
        with mock.patch.object(config, 'getbool', autospec=True) as mock_getbool:
            with mock.patch.object(config, 'items', autospec=True) as mock_items:
                mock_get.side_effect = _config_get_side_effect
                mock_getbool.side_effect = _config_get_side_effect
                mock_items.return_value = boto_config_dict.items()
                yield