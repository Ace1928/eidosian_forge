import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
class Dynamizer(object):
    """Control serialization/deserialization of types.

    This class controls the encoding of python types to the
    format that is expected by the DynamoDB API, as well as
    taking DynamoDB types and constructing the appropriate
    python types.

    If you want to customize this process, you can subclass
    this class and override the encoding/decoding of
    specific types.  For example::

        'foo'      (Python type)
            |
            v
        encode('foo')
            |
            v
        _encode_s('foo')
            |
            v
        {'S': 'foo'}  (Encoding sent to/received from DynamoDB)
            |
            V
        decode({'S': 'foo'})
            |
            v
        _decode_s({'S': 'foo'})
            |
            v
        'foo'     (Python type)

    """

    def _get_dynamodb_type(self, attr):
        return get_dynamodb_type(attr)

    def encode(self, attr):
        """
        Encodes a python type to the format expected
        by DynamoDB.

        """
        dynamodb_type = self._get_dynamodb_type(attr)
        try:
            encoder = getattr(self, '_encode_%s' % dynamodb_type.lower())
        except AttributeError:
            raise ValueError('Unable to encode dynamodb type: %s' % dynamodb_type)
        return {dynamodb_type: encoder(attr)}

    def _encode_n(self, attr):
        try:
            if isinstance(attr, float) and (not hasattr(Decimal, 'from_float')):
                n = str(float_to_decimal(attr))
            else:
                n = str(DYNAMODB_CONTEXT.create_decimal(attr))
            if list(filter(lambda x: x in n, ('Infinity', 'NaN'))):
                raise TypeError('Infinity and NaN not supported')
            return n
        except (TypeError, DecimalException) as e:
            msg = '{0} numeric for `{1}`\n{2}'.format(e.__class__.__name__, attr, str(e) or '')
        raise DynamoDBNumberError(msg)

    def _encode_s(self, attr):
        if isinstance(attr, bytes):
            attr = attr.decode('utf-8')
        elif not isinstance(attr, six.text_type):
            attr = str(attr)
        return attr

    def _encode_ns(self, attr):
        return list(map(self._encode_n, attr))

    def _encode_ss(self, attr):
        return [self._encode_s(n) for n in attr]

    def _encode_b(self, attr):
        if isinstance(attr, bytes):
            attr = Binary(attr)
        return attr.encode()

    def _encode_bs(self, attr):
        return [self._encode_b(n) for n in attr]

    def _encode_null(self, attr):
        return True

    def _encode_bool(self, attr):
        return attr

    def _encode_m(self, attr):
        return dict([(k, self.encode(v)) for k, v in attr.items()])

    def _encode_l(self, attr):
        return [self.encode(i) for i in attr]

    def decode(self, attr):
        """
        Takes the format returned by DynamoDB and constructs
        the appropriate python type.

        """
        if len(attr) > 1 or not attr or is_str(attr):
            return attr
        dynamodb_type = list(attr.keys())[0]
        if dynamodb_type.lower() == dynamodb_type:
            return attr
        try:
            decoder = getattr(self, '_decode_%s' % dynamodb_type.lower())
        except AttributeError:
            return attr
        return decoder(attr[dynamodb_type])

    def _decode_n(self, attr):
        return DYNAMODB_CONTEXT.create_decimal(attr)

    def _decode_s(self, attr):
        return attr

    def _decode_ns(self, attr):
        return set(map(self._decode_n, attr))

    def _decode_ss(self, attr):
        return set(map(self._decode_s, attr))

    def _decode_b(self, attr):
        return convert_binary(attr)

    def _decode_bs(self, attr):
        return set(map(self._decode_b, attr))

    def _decode_null(self, attr):
        return None

    def _decode_bool(self, attr):
        return attr

    def _decode_m(self, attr):
        return dict([(k, self.decode(v)) for k, v in attr.items()])

    def _decode_l(self, attr):
        return [self.decode(i) for i in attr]