import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
class TestRegionInfo(object):

    def __init__(self, connection=None, name=None, endpoint=None, connection_cls=None):
        self.connection = connection
        self.name = name
        self.endpoint = endpoint
        self.connection_cls = connection_cls

    def connect(self, **kwargs):
        return self.connection_cls(region=self)