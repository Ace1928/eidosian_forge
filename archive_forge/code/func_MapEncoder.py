import struct
from cloudsdk.google.protobuf.internal import wire_format
def MapEncoder(field_descriptor):
    """Encoder for extensions of MessageSet.

  Maps always have a wire format like this:
    message MapEntry {
      key_type key = 1;
      value_type value = 2;
    }
    repeated MapEntry map = N;
  """
    message_type = field_descriptor.message_type
    encode_message = MessageEncoder(field_descriptor.number, False, False)

    def EncodeField(write, value, deterministic):
        value_keys = sorted(value.keys()) if deterministic else value
        for key in value_keys:
            entry_msg = message_type._concrete_class(key=key, value=value[key])
            encode_message(write, entry_msg, deterministic)
    return EncodeField