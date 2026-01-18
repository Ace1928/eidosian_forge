from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def CancelApiCalls(self):
    if self.__default_stub:
        self.__default_stub.CancelApiCalls()