from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
class GeneratedProtocolMessageType(type):
    """Metaclass for protocol message classes created at runtime from Descriptors.

  We add implementations for all methods described in the Message class.  We
  also create properties to allow getting/setting all fields in the protocol
  message.  Finally, we create slots to prevent users from accidentally
  "setting" nonexistent fields in the protocol message, which then wouldn't get
  serialized / deserialized properly.

  The protocol compiler currently uses this metaclass to create protocol
  message classes at runtime.  Clients can also manually create their own
  classes at runtime, as in this example:

  mydescriptor = Descriptor(.....)
  factory = symbol_database.Default()
  factory.pool.AddDescriptor(mydescriptor)
  MyProtoClass = factory.GetPrototype(mydescriptor)
  myproto_instance = MyProtoClass()
  myproto.foo_field = 23
  ...
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'

    def __new__(cls, name, bases, dictionary):
        """Custom allocation for runtime-generated class types.

    We override __new__ because this is apparently the only place
    where we can meaningfully set __slots__ on the class we're creating(?).
    (The interplay between metaclasses and slots is not very well-documented).

    Args:
      name: Name of the class (ignored, but required by the
        metaclass protocol).
      bases: Base classes of the class we're constructing.
        (Should be message.Message).  We ignore this field, but
        it's required by the metaclass protocol
      dictionary: The class dictionary of the class we're
        constructing.  dictionary[_DESCRIPTOR_KEY] must contain
        a Descriptor object describing this protocol message
        type.

    Returns:
      Newly-allocated class.

    Raises:
      RuntimeError: Generated code only work with python cpp extension.
    """
        descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]
        if isinstance(descriptor, str):
            raise RuntimeError('The generated code only work with python cpp extension, but it is using pure python runtime.')
        new_class = getattr(descriptor, '_concrete_class', None)
        if new_class:
            return new_class
        if descriptor.full_name in well_known_types.WKTBASES:
            bases += (well_known_types.WKTBASES[descriptor.full_name],)
        _AddClassAttributesForNestedExtensions(descriptor, dictionary)
        _AddSlots(descriptor, dictionary)
        superclass = super(GeneratedProtocolMessageType, cls)
        new_class = superclass.__new__(cls, name, bases, dictionary)
        return new_class

    def __init__(cls, name, bases, dictionary):
        """Here we perform the majority of our work on the class.
    We add enum getters, an __init__ method, implementations
    of all Message methods, and properties for all fields
    in the protocol type.

    Args:
      name: Name of the class (ignored, but required by the
        metaclass protocol).
      bases: Base classes of the class we're constructing.
        (Should be message.Message).  We ignore this field, but
        it's required by the metaclass protocol
      dictionary: The class dictionary of the class we're
        constructing.  dictionary[_DESCRIPTOR_KEY] must contain
        a Descriptor object describing this protocol message
        type.
    """
        descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]
        existing_class = getattr(descriptor, '_concrete_class', None)
        if existing_class:
            assert existing_class is cls, 'Duplicate `GeneratedProtocolMessageType` created for descriptor %r' % descriptor.full_name
            return
        cls._message_set_decoders_by_tag = {}
        cls._fields_by_tag = {}
        if descriptor.has_options and descriptor.GetOptions().message_set_wire_format:
            cls._message_set_decoders_by_tag[decoder.MESSAGE_SET_ITEM_TAG] = (decoder.MessageSetItemDecoder(descriptor), None)
        for field in descriptor.fields:
            _AttachFieldHelpers(cls, field)
        if descriptor.is_extendable and hasattr(descriptor.file, 'pool'):
            extensions = descriptor.file.pool.FindAllExtensions(descriptor)
            for ext in extensions:
                _AttachFieldHelpers(cls, ext)
        descriptor._concrete_class = cls
        _AddEnumValues(descriptor, cls)
        _AddInitMethod(descriptor, cls)
        _AddPropertiesForFields(descriptor, cls)
        _AddPropertiesForExtensions(descriptor, cls)
        _AddStaticMethods(cls)
        _AddMessageMethods(descriptor, cls)
        _AddPrivateHelperMethods(descriptor, cls)
        superclass = super(GeneratedProtocolMessageType, cls)
        superclass.__init__(name, bases, dictionary)