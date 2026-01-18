from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import python_message
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
def BuildNestedDescriptors(msg_des, prefix):
    for name, nested_msg in msg_des.nested_types_by_name.items():
        module_name = prefix + name.upper()
        module[module_name] = nested_msg
        BuildNestedDescriptors(nested_msg, module_name + '_')
    for enum_des in msg_des.enum_types:
        module[prefix + enum_des.name.upper()] = enum_des