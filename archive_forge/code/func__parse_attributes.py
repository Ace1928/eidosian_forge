import datetime
from google.api_core.exceptions import InvalidArgument
from cloudsdk.google.protobuf.timestamp_pb2 import Timestamp  # pytype: disable=pyi-error
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub import MessageTransformer
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1 import AttributeValues, SequencedMessage, PubSubMessage
def _parse_attributes(values: AttributeValues) -> str:
    if not len(values.values) == 1:
        raise InvalidArgument('Received an unparseable message with multiple values for an attribute.')
    value: bytes = values.values[0]
    try:
        return value.decode('utf-8')
    except UnicodeError:
        raise InvalidArgument('Received an unparseable message with a non-utf8 attribute.')