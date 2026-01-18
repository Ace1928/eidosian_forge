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
class _MessageInfo:
    """Metadata about a message.

    Args:
        fields (Tuple[~.fields.Field]): The fields declared on the message.
        package (str): The proto package.
        full_name (str): The full name of the message.
        file_info (~._FileInfo): The file descriptor and messages for the
            file containing this message.
        marshal (~.Marshal): The marshal instance to which this message was
            automatically registered.
        options (~.descriptor_pb2.MessageOptions): Any options that were
            set on the message.
    """

    def __init__(self, *, fields: List[Field], package: str, full_name: str, marshal: Marshal, options: descriptor_pb2.MessageOptions) -> None:
        self.package = package
        self.full_name = full_name
        self.options = options
        self.fields = collections.OrderedDict(((i.name, i) for i in fields))
        self.fields_by_number = collections.OrderedDict(((i.number, i) for i in fields))
        self.marshal = marshal
        self._pb = None

    @property
    def pb(self) -> Type[message.Message]:
        """Return the protobuf message type for this descriptor.

        If a field on the message references another message which has not
        loaded, then this method returns None.
        """
        return self._pb