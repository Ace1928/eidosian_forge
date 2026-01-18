import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _MergeMessage(node, source, destination, replace_message, replace_repeated):
    """Merge all fields specified by a sub-tree from source to destination."""
    source_descriptor = source.DESCRIPTOR
    for name in node:
        child = node[name]
        field = source_descriptor.fields_by_name[name]
        if field is None:
            raise ValueError("Error: Can't find field {0} in message {1}.".format(name, source_descriptor.full_name))
        if child:
            if field.label == FieldDescriptor.LABEL_REPEATED or field.cpp_type != FieldDescriptor.CPPTYPE_MESSAGE:
                raise ValueError('Error: Field {0} in message {1} is not a singular message field and cannot have sub-fields.'.format(name, source_descriptor.full_name))
            if source.HasField(name):
                _MergeMessage(child, getattr(source, name), getattr(destination, name), replace_message, replace_repeated)
            continue
        if field.label == FieldDescriptor.LABEL_REPEATED:
            if replace_repeated:
                destination.ClearField(_StrConvert(name))
            repeated_source = getattr(source, name)
            repeated_destination = getattr(destination, name)
            repeated_destination.MergeFrom(repeated_source)
        elif field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
            if replace_message:
                destination.ClearField(_StrConvert(name))
            if source.HasField(name):
                getattr(destination, name).MergeFrom(getattr(source, name))
        else:
            setattr(destination, name, getattr(source, name))