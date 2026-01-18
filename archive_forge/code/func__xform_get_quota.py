import argparse
import itertools
import logging
import sys
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _xform_get_quota(data, value, keys):
    res = []
    res_info = {}
    for key in keys:
        res_info[key] = getattr(data, key, '')
    res_info['id'] = value
    res.append(res_info)
    return res