from collections import OrderedDict
import hashlib
import os
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message_factory
def _MakeFileDescriptorProto(proto_file_name, full_name, field_items):
    """Populate FileDescriptorProto for MessageFactory's DescriptorPool."""
    package, name = full_name.rsplit('.', 1)
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = os.path.join(package.replace('.', '/'), proto_file_name)
    file_proto.package = package
    desc_proto = file_proto.message_type.add()
    desc_proto.name = name
    for f_number, (f_name, f_type) in enumerate(field_items, 1):
        field_proto = desc_proto.field.add()
        field_proto.name = f_name
        if f_number >= descriptor.FieldDescriptor.FIRST_RESERVED_FIELD_NUMBER:
            f_number += descriptor.FieldDescriptor.LAST_RESERVED_FIELD_NUMBER - descriptor.FieldDescriptor.FIRST_RESERVED_FIELD_NUMBER + 1
        field_proto.number = f_number
        field_proto.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field_proto.type = f_type
    return file_proto