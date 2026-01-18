import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def _check_deprecation(self, func, argv):
    if not hasattr(func, 'deprecated_groups'):
        return
    for old_info, new_info, required in func.deprecated_groups:
        old_param = old_info[0][0]
        new_param = new_info[0][0]
        old_value, new_value = (None, None)
        for i in range(len(argv)):
            cur_arg = argv[i]
            if cur_arg == old_param:
                old_value = argv[i + 1]
            elif cur_arg == new_param[0]:
                new_value = argv[i + 1]
        if old_value and (not new_value):
            print('WARNING: The %s parameter is deprecated and will be removed in a future release. Use the %s parameter to avoid seeing this message.' % (old_param, new_param))