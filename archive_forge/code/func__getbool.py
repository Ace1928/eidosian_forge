import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def _getbool(section, name, default=False):
    if section == 'Boto' and name == 'use_endpoint_heuristics':
        return True
    return default