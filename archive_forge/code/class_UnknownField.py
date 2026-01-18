from google.protobuf.internal import api_implementation
class UnknownField:
    """A parsed unknown field."""
    __slots__ = ['_field_number', '_wire_type', '_data']

    def __init__(self, field_number, wire_type, data):
        self._field_number = field_number
        self._wire_type = wire_type
        self._data = data
        return

    @property
    def field_number(self):
        return self._field_number

    @property
    def wire_type(self):
        return self._wire_type

    @property
    def data(self):
        return self._data