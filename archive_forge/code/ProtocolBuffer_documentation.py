from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
Serialize a range of extensions.

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
    