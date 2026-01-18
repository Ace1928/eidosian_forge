import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class FieldDescriptor(DescriptorBase):
    """Descriptor for a single field in a .proto file.

  Attributes:
    name (str): Name of this field, exactly as it appears in .proto.
    full_name (str): Name of this field, including containing scope.  This is
      particularly relevant for extensions.
    index (int): Dense, 0-indexed index giving the order that this
      field textually appears within its message in the .proto file.
    number (int): Tag number declared for this field in the .proto file.

    type (int): (One of the TYPE_* constants below) Declared type.
    cpp_type (int): (One of the CPPTYPE_* constants below) C++ type used to
      represent this field.

    label (int): (One of the LABEL_* constants below) Tells whether this
      field is optional, required, or repeated.
    has_default_value (bool): True if this field has a default value defined,
      otherwise false.
    default_value (Varies): Default value of this field.  Only
      meaningful for non-repeated scalar fields.  Repeated fields
      should always set this to [], and non-repeated composite
      fields should always set this to None.

    containing_type (Descriptor): Descriptor of the protocol message
      type that contains this field.  Set by the Descriptor constructor
      if we're passed into one.
      Somewhat confusingly, for extension fields, this is the
      descriptor of the EXTENDED message, not the descriptor
      of the message containing this field.  (See is_extension and
      extension_scope below).
    message_type (Descriptor): If a composite field, a descriptor
      of the message type contained in this field.  Otherwise, this is None.
    enum_type (EnumDescriptor): If this field contains an enum, a
      descriptor of that enum.  Otherwise, this is None.

    is_extension: True iff this describes an extension field.
    extension_scope (Descriptor): Only meaningful if is_extension is True.
      Gives the message that immediately contains this extension field.
      Will be None iff we're a top-level (file-level) extension field.

    options (descriptor_pb2.FieldOptions): Protocol message field options or
      None to use default field options.

    containing_oneof (OneofDescriptor): If the field is a member of a oneof
      union, contains its descriptor. Otherwise, None.

    file (FileDescriptor): Reference to file descriptor.
  """
    TYPE_DOUBLE = 1
    TYPE_FLOAT = 2
    TYPE_INT64 = 3
    TYPE_UINT64 = 4
    TYPE_INT32 = 5
    TYPE_FIXED64 = 6
    TYPE_FIXED32 = 7
    TYPE_BOOL = 8
    TYPE_STRING = 9
    TYPE_GROUP = 10
    TYPE_MESSAGE = 11
    TYPE_BYTES = 12
    TYPE_UINT32 = 13
    TYPE_ENUM = 14
    TYPE_SFIXED32 = 15
    TYPE_SFIXED64 = 16
    TYPE_SINT32 = 17
    TYPE_SINT64 = 18
    MAX_TYPE = 18
    CPPTYPE_INT32 = 1
    CPPTYPE_INT64 = 2
    CPPTYPE_UINT32 = 3
    CPPTYPE_UINT64 = 4
    CPPTYPE_DOUBLE = 5
    CPPTYPE_FLOAT = 6
    CPPTYPE_BOOL = 7
    CPPTYPE_ENUM = 8
    CPPTYPE_STRING = 9
    CPPTYPE_MESSAGE = 10
    MAX_CPPTYPE = 10
    _PYTHON_TO_CPP_PROTO_TYPE_MAP = {TYPE_DOUBLE: CPPTYPE_DOUBLE, TYPE_FLOAT: CPPTYPE_FLOAT, TYPE_ENUM: CPPTYPE_ENUM, TYPE_INT64: CPPTYPE_INT64, TYPE_SINT64: CPPTYPE_INT64, TYPE_SFIXED64: CPPTYPE_INT64, TYPE_UINT64: CPPTYPE_UINT64, TYPE_FIXED64: CPPTYPE_UINT64, TYPE_INT32: CPPTYPE_INT32, TYPE_SFIXED32: CPPTYPE_INT32, TYPE_SINT32: CPPTYPE_INT32, TYPE_UINT32: CPPTYPE_UINT32, TYPE_FIXED32: CPPTYPE_UINT32, TYPE_BYTES: CPPTYPE_STRING, TYPE_STRING: CPPTYPE_STRING, TYPE_BOOL: CPPTYPE_BOOL, TYPE_MESSAGE: CPPTYPE_MESSAGE, TYPE_GROUP: CPPTYPE_MESSAGE}
    LABEL_OPTIONAL = 1
    LABEL_REQUIRED = 2
    LABEL_REPEATED = 3
    MAX_LABEL = 3
    MAX_FIELD_NUMBER = (1 << 29) - 1
    FIRST_RESERVED_FIELD_NUMBER = 19000
    LAST_RESERVED_FIELD_NUMBER = 19999
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.FieldDescriptor

        def __new__(cls, name, full_name, index, number, type, cpp_type, label, default_value, message_type, enum_type, containing_type, is_extension, extension_scope, options=None, serialized_options=None, has_default_value=True, containing_oneof=None, json_name=None, file=None, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            if is_extension:
                return _message.default_pool.FindExtensionByName(full_name)
            else:
                return _message.default_pool.FindFieldByName(full_name)

    def __init__(self, name, full_name, index, number, type, cpp_type, label, default_value, message_type, enum_type, containing_type, is_extension, extension_scope, options=None, serialized_options=None, has_default_value=True, containing_oneof=None, json_name=None, file=None, create_key=None):
        """The arguments are as described in the description of FieldDescriptor
    attributes above.

    Note that containing_type may be None, and may be set later if necessary
    (to deal with circular references between message types, for example).
    Likewise for extension_scope.
    """
        if create_key is not _internal_create_key:
            _Deprecated('FieldDescriptor')
        super(FieldDescriptor, self).__init__(file, options, serialized_options, 'FieldOptions')
        self.name = name
        self.full_name = full_name
        self._camelcase_name = None
        if json_name is None:
            self.json_name = _ToJsonName(name)
        else:
            self.json_name = json_name
        self.index = index
        self.number = number
        self._type = type
        self.cpp_type = cpp_type
        self._label = label
        self.has_default_value = has_default_value
        self.default_value = default_value
        self.containing_type = containing_type
        self.message_type = message_type
        self.enum_type = enum_type
        self.is_extension = is_extension
        self.extension_scope = extension_scope
        self.containing_oneof = containing_oneof
        if api_implementation.Type() == 'python':
            self._cdescriptor = None
        elif is_extension:
            self._cdescriptor = _message.default_pool.FindExtensionByName(full_name)
        else:
            self._cdescriptor = _message.default_pool.FindFieldByName(full_name)

    @property
    def _parent(self):
        if self.containing_oneof:
            return self.containing_oneof
        if self.is_extension:
            return self.extension_scope or self.file
        return self.containing_type

    def _InferLegacyFeatures(self, edition, options, features):
        from google.protobuf import descriptor_pb2
        if edition >= descriptor_pb2.Edition.EDITION_2023:
            return
        if self._label == FieldDescriptor.LABEL_REQUIRED:
            features.field_presence = descriptor_pb2.FeatureSet.FieldPresence.LEGACY_REQUIRED
        if self._type == FieldDescriptor.TYPE_GROUP:
            features.message_encoding = descriptor_pb2.FeatureSet.MessageEncoding.DELIMITED
        if options.HasField('packed'):
            features.repeated_field_encoding = descriptor_pb2.FeatureSet.RepeatedFieldEncoding.PACKED if options.packed else descriptor_pb2.FeatureSet.RepeatedFieldEncoding.EXPANDED

    @property
    def type(self):
        if self._GetFeatures().message_encoding == _FEATURESET_MESSAGE_ENCODING_DELIMITED:
            return FieldDescriptor.TYPE_GROUP
        return self._type

    @type.setter
    def type(self, val):
        self._type = val

    @property
    def label(self):
        if self._GetFeatures().field_presence == _FEATURESET_FIELD_PRESENCE_LEGACY_REQUIRED:
            return FieldDescriptor.LABEL_REQUIRED
        return self._label

    @property
    def camelcase_name(self):
        """Camelcase name of this field.

    Returns:
      str: the name in CamelCase.
    """
        if self._camelcase_name is None:
            self._camelcase_name = _ToCamelCase(self.name)
        return self._camelcase_name

    @property
    def has_presence(self):
        """Whether the field distinguishes between unpopulated and default values.

    Raises:
      RuntimeError: singular field that is not linked with message nor file.
    """
        if self.label == FieldDescriptor.LABEL_REPEATED:
            return False
        if self.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE or self.containing_oneof:
            return True
        return self._GetFeatures().field_presence != _FEATURESET_FIELD_PRESENCE_IMPLICIT

    @property
    def is_packed(self):
        """Returns if the field is packed."""
        if self.label != FieldDescriptor.LABEL_REPEATED:
            return False
        field_type = self.type
        if field_type == FieldDescriptor.TYPE_STRING or field_type == FieldDescriptor.TYPE_GROUP or field_type == FieldDescriptor.TYPE_MESSAGE or (field_type == FieldDescriptor.TYPE_BYTES):
            return False
        return self._GetFeatures().repeated_field_encoding == _FEATURESET_REPEATED_FIELD_ENCODING_PACKED

    @staticmethod
    def ProtoTypeToCppProtoType(proto_type):
        """Converts from a Python proto type to a C++ Proto Type.

    The Python ProtocolBuffer classes specify both the 'Python' datatype and the
    'C++' datatype - and they're not the same. This helper method should
    translate from one to another.

    Args:
      proto_type: the Python proto type (descriptor.FieldDescriptor.TYPE_*)
    Returns:
      int: descriptor.FieldDescriptor.CPPTYPE_*, the C++ type.
    Raises:
      TypeTransformationError: when the Python proto type isn't known.
    """
        try:
            return FieldDescriptor._PYTHON_TO_CPP_PROTO_TYPE_MAP[proto_type]
        except KeyError:
            raise TypeTransformationError('Unknown proto_type: %s' % proto_type)