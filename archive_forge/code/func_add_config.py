from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def add_config(self):
    x = CapabilityConfig()
    self.config_.append(x)
    return x