from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
from collections import namedtuple
import six
from apitools.base.protorpclite import protojson
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def IsEqualBindings(a, b):
    granted, removed = DiffBindings(a, b)
    return not granted.bindings and (not removed.bindings)