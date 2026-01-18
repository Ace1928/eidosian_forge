import warnings
from google.protobuf.internal import api_implementation
from google.protobuf import descriptor_pool
from google.protobuf import message
def _InternalCreateMessageClass(descriptor):
    """Builds a proto2 message class based on the passed in descriptor.

  Args:
    descriptor: The descriptor to build from.

  Returns:
    A class describing the passed in descriptor.
  """
    descriptor_name = descriptor.name
    result_class = _GENERATED_PROTOCOL_MESSAGE_TYPE(descriptor_name, (message.Message,), {'DESCRIPTOR': descriptor, '__module__': None})
    for field in descriptor.fields:
        if field.message_type:
            GetMessageClass(field.message_type)
    for extension in result_class.DESCRIPTOR.extensions:
        extended_class = GetMessageClass(extension.containing_type)
        if api_implementation.Type() != 'python':
            pool = extension.containing_type.file.pool
            if extension is not pool.FindExtensionByNumber(extension.containing_type, extension.number):
                raise ValueError('Double registration of Extensions')
        if extension.message_type:
            GetMessageClass(extension.message_type)
    return result_class