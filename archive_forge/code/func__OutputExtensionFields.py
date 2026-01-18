from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _OutputExtensionFields(self, out, partial, extensions, start_index, end_field_number):
    """Serialize a range of extensions.

    To generate canonical output when encoding, we interleave fields and
    extensions to preserve tag order.

    Generated code will prepare a list of ExtensionIdentifier sorted in field
    number order and call this method to serialize a specific range of
    extensions. The range is specified by the two arguments, start_index and
    end_field_number.

    The method will serialize all extensions[i] with i >= start_index and
    extensions[i].number < end_field_number. Since extensions argument is sorted
    by field_number, this is a contiguous range; the first index j not included
    in that range is returned. The return value can be used as the start_index
    in the next call to serialize the next range of extensions.

    Args:
      extensions: A list of ExtensionIdentifier sorted in field number order.
      start_index: The start index in the extensions list.
      end_field_number: The end field number of the extension range.

    Returns:
      The first index that is not in the range. Or the size of extensions if all
      the extensions are within the range.
    """

    def OutputSingleField(ext, value):
        out.putVarInt32(ext.wire_tag)
        if ext.field_type == TYPE_GROUP:
            if partial:
                value.OutputPartial(out)
            else:
                value.OutputUnchecked(out)
            out.putVarInt32(ext.wire_tag + 1)
        elif ext.field_type == TYPE_FOREIGN:
            if partial:
                out.putVarInt32(value.ByteSizePartial())
                value.OutputPartial(out)
            else:
                out.putVarInt32(value.ByteSize())
                value.OutputUnchecked(out)
        else:
            Encoder._TYPE_TO_METHOD[ext.field_type](out, value)
    for ext_index, ext in enumerate(itertools.islice(extensions, start_index, None), start=start_index):
        if ext.number >= end_field_number:
            return ext_index
        if ext.is_repeated:
            for field in self._extension_fields[ext]:
                OutputSingleField(ext, field)
        else:
            OutputSingleField(ext, self._extension_fields[ext])
    return len(extensions)