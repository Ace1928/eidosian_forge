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
def DeserializeBindingsTuple(serialized_bindings_tuple):
    is_grant, bindings = serialized_bindings_tuple
    return BindingsTuple(is_grant=is_grant, bindings=[protojson.decode_message(apitools_messages.Policy.BindingsValueListEntry, t) for t in bindings])