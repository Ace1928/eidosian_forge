from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import python_message
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
def BuildTopDescriptorsAndMessages(file_des, module_name, module):
    """Builds top level descriptors and message classes.

  Args:
    file_des: FileDescriptor of the .proto file
    module_name: str, the name of generated _pb2 module
    module: Generated _pb2 module
  """

    def BuildMessage(msg_des):
        create_dict = {}
        for name, nested_msg in msg_des.nested_types_by_name.items():
            create_dict[name] = BuildMessage(nested_msg)
        create_dict['DESCRIPTOR'] = msg_des
        create_dict['__module__'] = module_name
        message_class = _reflection.GeneratedProtocolMessageType(msg_des.name, (_message.Message,), create_dict)
        _sym_db.RegisterMessage(message_class)
        return message_class
    for name, enum_des in file_des.enum_types_by_name.items():
        module['_' + name.upper()] = enum_des
        module[name] = enum_type_wrapper.EnumTypeWrapper(enum_des)
        for enum_value in enum_des.values:
            module[enum_value.name] = enum_value.number
    for name, extension_des in file_des.extensions_by_name.items():
        module[name.upper() + '_FIELD_NUMBER'] = extension_des.number
        module[name] = extension_des
    for name, service in file_des.services_by_name.items():
        module['_' + name.upper()] = service
    for name, msg_des in file_des.message_types_by_name.items():
        module[name] = BuildMessage(msg_des)