import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def LookupDescriptorOrDie(self, name):
    message_descriptor = self.LookupDescriptor(name)
    if message_descriptor is None:
        raise ValueError('No message descriptor named "%s"' % name)
    return message_descriptor