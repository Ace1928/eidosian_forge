import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
def _get_remaining_manifest(self, new_class):
    return self._get_manifest(new_class) - {new_class.__name__}