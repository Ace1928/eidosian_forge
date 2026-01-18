from cloudsdk.google.protobuf.internal import type_checkers
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
class _ExtensionDict(object):
    """Dict-like container for Extension fields on proto instances.

  Note that in all cases we expect extension handles to be
  FieldDescriptors.
  """

    def __init__(self, extended_message):
        """
    Args:
      extended_message: Message instance for which we are the Extensions dict.
    """
        self._extended_message = extended_message

    def __getitem__(self, extension_handle):
        """Returns the current value of the given extension handle."""
        _VerifyExtensionHandle(self._extended_message, extension_handle)
        result = self._extended_message._fields.get(extension_handle)
        if result is not None:
            return result
        if extension_handle.label == FieldDescriptor.LABEL_REPEATED:
            result = extension_handle._default_constructor(self._extended_message)
        elif extension_handle.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
            message_type = extension_handle.message_type
            if not hasattr(message_type, '_concrete_class'):
                self._extended_message._FACTORY.GetPrototype(message_type)
            assert getattr(extension_handle.message_type, '_concrete_class', None), 'Uninitialized concrete class found for field %r (message type %r)' % (extension_handle.full_name, extension_handle.message_type.full_name)
            result = extension_handle.message_type._concrete_class()
            try:
                result._SetListener(self._extended_message._listener_for_children)
            except ReferenceError:
                pass
        else:
            return extension_handle.default_value
        result = self._extended_message._fields.setdefault(extension_handle, result)
        return result

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        my_fields = self._extended_message.ListFields()
        other_fields = other._extended_message.ListFields()
        my_fields = [field for field in my_fields if field.is_extension]
        other_fields = [field for field in other_fields if field.is_extension]
        return my_fields == other_fields

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        fields = self._extended_message.ListFields()
        extension_fields = [field for field in fields if field[0].is_extension]
        return len(extension_fields)

    def __hash__(self):
        raise TypeError('unhashable object')

    def __setitem__(self, extension_handle, value):
        """If extension_handle specifies a non-repeated, scalar extension
    field, sets the value of that field.
    """
        _VerifyExtensionHandle(self._extended_message, extension_handle)
        if extension_handle.label == FieldDescriptor.LABEL_REPEATED or extension_handle.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
            raise TypeError('Cannot assign to extension "%s" because it is a repeated or composite type.' % extension_handle.full_name)
        type_checker = type_checkers.GetTypeChecker(extension_handle)
        self._extended_message._fields[extension_handle] = type_checker.CheckValue(value)
        self._extended_message._Modified()

    def __delitem__(self, extension_handle):
        self._extended_message.ClearExtension(extension_handle)

    def _FindExtensionByName(self, name):
        """Tries to find a known extension with the specified name.

    Args:
      name: Extension full name.

    Returns:
      Extension field descriptor.
    """
        return self._extended_message._extensions_by_name.get(name, None)

    def _FindExtensionByNumber(self, number):
        """Tries to find a known extension with the field number.

    Args:
      number: Extension field number.

    Returns:
      Extension field descriptor.
    """
        return self._extended_message._extensions_by_number.get(number, None)

    def __iter__(self):
        return (f[0] for f in self._extended_message.ListFields() if f[0].is_extension)

    def __contains__(self, extension_handle):
        _VerifyExtensionHandle(self._extended_message, extension_handle)
        if extension_handle not in self._extended_message._fields:
            return False
        if extension_handle.label == FieldDescriptor.LABEL_REPEATED:
            return bool(self._extended_message._fields.get(extension_handle))
        if extension_handle.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
            value = self._extended_message._fields.get(extension_handle)
            return value is not None and value._is_present_in_parent
        return True