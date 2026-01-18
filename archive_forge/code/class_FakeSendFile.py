import os
import routes
import webob
from glance.api.middleware import context
from glance.api.v2 import router
import glance.common.client
class FakeSendFile(object):

    def __init__(self, req):
        self.req = req

    def sendfile(self, o, i, offset, nbytes):
        os.lseek(i, offset, os.SEEK_SET)
        prev_len = len(self.req.body)
        self.req.body += os.read(i, nbytes)
        return len(self.req.body) - prev_len