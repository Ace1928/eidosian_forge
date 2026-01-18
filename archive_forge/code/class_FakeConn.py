import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
class FakeConn(object):

    def __init__(self, region, **kwargs):
        self.region = region
        self.kwargs = kwargs