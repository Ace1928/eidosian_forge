from __future__ import absolute_import
import io
import logging
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import AbstractRpcServer
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import HttpRpcServer
from googlecloudsdk.third_party.appengine._internal import six_subset
class UrlLibRequestResponseStub(object):

    def __init__(self, headers=None):
        self.headers = {}
        if headers:
            self.headers = headers

    def add_header(self, header, value):
        self.headers[header] = value