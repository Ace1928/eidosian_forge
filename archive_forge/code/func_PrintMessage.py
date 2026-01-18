import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re
from google.protobuf.internal import decoder
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
from google.protobuf import unknown_fields
def PrintMessage(self, message):
    """Convert protobuf message to text format.

    Args:
      message: The protocol buffers message.
    """
    if self.message_formatter and self._TryCustomFormatMessage(message):
        return
    if message.DESCRIPTOR.full_name == _ANY_FULL_TYPE_NAME and self._TryPrintAsAnyMessage(message):
        return
    fields = message.ListFields()
    if self.use_index_order:
        fields.sort(key=lambda x: x[0].number if x[0].is_extension else x[0].index)
    for field, value in fields:
        if _IsMapEntry(field):
            for key in sorted(value):
                entry_submsg = value.GetEntryClass()(key=key, value=value[key])
                self.PrintField(field, entry_submsg)
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            if self.use_short_repeated_primitives and field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_MESSAGE and (field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_STRING):
                self._PrintShortRepeatedPrimitivesValue(field, value)
            else:
                for element in value:
                    self.PrintField(field, element)
        else:
            self.PrintField(field, value)
    if self.print_unknown_fields:
        self._PrintUnknownFields(unknown_fields.UnknownFieldSet(message))