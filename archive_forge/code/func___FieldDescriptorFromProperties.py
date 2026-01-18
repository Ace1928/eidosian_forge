import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __FieldDescriptorFromProperties(self, name, index, attrs):
    """Create a field descriptor for these attrs."""
    field = descriptor.FieldDescriptor()
    field.name = self.__names.CleanName(name)
    field.number = index
    field.label = self.__ComputeLabel(attrs)
    new_type_name_hint = self.__names.ClassName('%sValue' % self.__names.ClassName(name))
    type_info = self.__GetTypeInfo(attrs, new_type_name_hint)
    field.type_name = type_info.type_name
    field.variant = type_info.variant
    if 'default' in attrs:
        default = attrs['default']
        if not (field.type_name == 'string' or field.variant == messages.Variant.ENUM):
            default = str(json.loads(default))
        if field.variant == messages.Variant.ENUM:
            default = self.__names.NormalizeEnumName(default)
        field.default_value = default
    extended_field = extended_descriptor.ExtendedFieldDescriptor()
    extended_field.name = field.name
    extended_field.description = util.CleanDescription(attrs.get('description', 'A %s attribute.' % field.type_name))
    extended_field.field_descriptor = field
    return extended_field