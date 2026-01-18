import base64
class BytesRule:
    """A marshal between Python strings and protobuf bytes.

    Note: this conversion is asymmetric because Python does have a bytes type.
    It is sometimes necessary to convert proto bytes fields to strings, e.g. for
    JSON encoding, marshalling a message to a dict. Because bytes fields can
    represent arbitrary data, bytes fields are base64 encoded when they need to
    be represented as strings.

    It is necessary to have the conversion be bidirectional, i.e.
    my_message == MyMessage(MyMessage.to_dict(my_message))

    To accomplish this, we need to intercept assignments from strings and
    base64 decode them back into bytes.
    """

    def to_python(self, value, *, absent: bool=None):
        return value

    def to_proto(self, value):
        if isinstance(value, str):
            value = value.encode('utf-8')
            value += b'=' * (4 - len(value) % 4)
            value = base64.urlsafe_b64decode(value)
        return value