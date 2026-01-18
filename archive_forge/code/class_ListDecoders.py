import re
from ovs.flow.kv import KeyValue, KeyMetadata, ParseError
from ovs.flow.decoders import decode_default
class ListDecoders(object):
    """ListDecoders is used by ListParser to decode the elements in the list.

    A decoder is a function that accepts a value and returns its decoded
    object.

    ListDecoders is initialized with a list of tuples that contains the
    keyword and the decoding function associated with each position in the
    list. The order is, therefore, important.

    Args:
        decoders (list of tuples): Optional; a list of tuples.
            The first element in the tuple is the keyword associated with the
            value. The second element in the tuple is the decoder function.
    """

    def __init__(self, decoders=None):
        self._decoders = decoders or list()

    def decode(self, index, value_str):
        """Decode the index'th element of the list.

        Args:
            index (int): The position in the list of the element to decode.
            value_str (str): The value string to decode.
        """
        if index < 0 or index >= len(self._decoders):
            if self._default_decoder:
                return self._default_decoder(index, value_str)
            else:
                raise ParseError(f'Cannot decode element {index} in list: {value_str}')
        try:
            key = self._decoders[index][0]
            value = self._decoders[index][1](value_str)
            return (key, value)
        except Exception as e:
            raise ParseError('Failed to decode value_str {}: {}'.format(value_str, str(e)))

    @staticmethod
    def _default_decoder(index, value):
        key = 'elem_{}'.format(index)
        return (key, decode_default(value))