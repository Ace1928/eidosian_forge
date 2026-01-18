import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def MessageToJson(message, preserving_proto_field_name=False, indent=2, sort_keys=False, use_integers_for_enums=False, descriptor_pool=None, float_precision=None, ensure_ascii=True, always_print_fields_with_no_presence=False):
    """Converts protobuf message to JSON format.

  Args:
    message: The protocol buffers message instance to serialize.
    always_print_fields_with_no_presence: If True, fields without
      presence (implicit presence scalars, repeated fields, and map fields) will
      always be serialized. Any field that supports presence is not affected by
      this option (including singular message fields and oneof fields).
    preserving_proto_field_name: If True, use the original proto field names as
      defined in the .proto file. If False, convert the field names to
      lowerCamelCase.
    indent: The JSON object will be pretty-printed with this indent level. An
      indent level of 0 or negative will only insert newlines. If the indent
      level is None, no newlines will be inserted.
    sort_keys: If True, then the output will be sorted by field names.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    float_precision: If set, use this to specify float field valid digits.
    ensure_ascii: If True, strings with non-ASCII characters are escaped. If
      False, Unicode strings are returned unchanged.

  Returns:
    A string containing the JSON formatted protocol buffer message.
  """
    printer = _Printer(preserving_proto_field_name, use_integers_for_enums, descriptor_pool, float_precision, always_print_fields_with_no_presence)
    return printer.ToJsonString(message, indent, sort_keys, ensure_ascii)