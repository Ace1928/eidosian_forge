import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
class TokenDecoder:
    """Decodes token strings back into dictionaries.

    This performs the inverse operation to the TokenEncoder, accepting
    opaque strings and decoding them into a useable form.
    """

    def decode(self, token):
        """Decodes an opaque string to a dictionary.

        :type token: str
        :param token: A token string given by the botocore pagination
            interface.

        :rtype: dict
        :returns: A dictionary containing pagination information,
            particularly the service pagination token(s) but also other boto
            metadata.
        """
        json_string = base64.b64decode(token.encode('utf-8')).decode('utf-8')
        decoded_token = json.loads(json_string)
        encoded_keys = decoded_token.pop('boto_encoded_keys', None)
        if encoded_keys is None:
            return decoded_token
        else:
            return self._decode(decoded_token, encoded_keys)

    def _decode(self, token, encoded_keys):
        """Find each encoded value and decode it."""
        for key in encoded_keys:
            encoded = self._path_get(token, key)
            decoded = base64.b64decode(encoded.encode('utf-8'))
            self._path_set(token, key, decoded)
        return token

    def _path_get(self, data, path):
        """Return the nested data at the given path.

        For instance:
            data = {'foo': ['bar', 'baz']}
            path = ['foo', 0]
            ==> 'bar'
        """
        d = data
        for step in path:
            d = d[step]
        return d

    def _path_set(self, data, path, value):
        """Set the value of a key in the given data.

        Example:
            data = {'foo': ['bar', 'baz']}
            path = ['foo', 1]
            value = 'bin'
            ==> data = {'foo': ['bar', 'bin']}
        """
        container = self._path_get(data, path[:-1])
        container[path[-1]] = value