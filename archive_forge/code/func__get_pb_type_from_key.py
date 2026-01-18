import collections
import collections.abc
import copy
import re
from typing import List, Type
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf.json_format import MessageToDict, MessageToJson, Parse
from proto import _file_info
from proto import _package_info
from proto.fields import Field
from proto.fields import MapField
from proto.fields import RepeatedField
from proto.marshal import Marshal
from proto.primitives import ProtoType
from proto.utils import has_upb
def _get_pb_type_from_key(self, key):
    """Given a key, return the corresponding pb_type.

        Args:
            key(str): The name of the field.

        Returns:
            A tuple containing a key and pb_type. The pb_type will be
            the composite type of the field, or the primitive type if a primitive.
            If no corresponding field exists, return None.
        """
    pb_type = None
    try:
        pb_type = self._meta.fields[key].pb_type
    except KeyError:
        if f'{key}_' in self._meta.fields:
            key = f'{key}_'
            pb_type = self._meta.fields[key].pb_type
    return (key, pb_type)