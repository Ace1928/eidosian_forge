from __future__ import absolute_import
import io
import logging
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import AbstractRpcServer
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import HttpRpcServer
from googlecloudsdk.third_party.appengine._internal import six_subset
class UrlLibResponseStub(UrlLibRequestResponseStub, io.BytesIO):

    def __init__(self, body, headers, url, code, msg):
        UrlLibRequestResponseStub.__init__(self, headers)
        if body:
            io.BytesIO.__init__(self, body)
        else:
            io.BytesIO.__init__(self, b'')
        self.url = url
        self.code = code
        self.msg = msg