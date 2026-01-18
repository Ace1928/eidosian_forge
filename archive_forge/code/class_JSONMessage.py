import base64
from boto.sqs.message import MHMessage
from boto.exception import SQSDecodeError
from boto.compat import json
class JSONMessage(MHMessage):
    """
    Acts like a dictionary but encodes it's data as a Base64 encoded JSON payload.
    """

    def decode(self, value):
        try:
            value = base64.b64decode(value.encode('utf-8')).decode('utf-8')
            value = json.loads(value)
        except:
            raise SQSDecodeError('Unable to decode message', self)
        return value

    def encode(self, value):
        value = json.dumps(value)
        return base64.b64encode(value.encode('utf-8')).decode('utf-8')