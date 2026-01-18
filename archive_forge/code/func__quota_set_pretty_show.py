from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def _quota_set_pretty_show(quotas):
    """Convert quotas object to dict and display."""
    new_quotas = {}
    for quota_k, quota_v in sorted(quotas.to_dict().items()):
        if isinstance(quota_v, dict):
            quota_v = '\n'.join(['%s = %s' % (k, v) for k, v in sorted(quota_v.items())])
        new_quotas[quota_k] = quota_v
    cliutils.print_dict(new_quotas)