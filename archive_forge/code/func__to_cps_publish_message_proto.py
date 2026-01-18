import datetime
from google.api_core.exceptions import InvalidArgument
from cloudsdk.google.protobuf.timestamp_pb2 import Timestamp  # pytype: disable=pyi-error
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub import MessageTransformer
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1 import AttributeValues, SequencedMessage, PubSubMessage
def _to_cps_publish_message_proto(source: PubSubMessage.meta.pb) -> PubsubMessage.meta.pb:
    out = PubsubMessage.meta.pb()
    try:
        out.ordering_key = source.key.decode('utf-8')
    except UnicodeError:
        raise InvalidArgument('Received an unparseable message with a non-utf8 key.')
    if PUBSUB_LITE_EVENT_TIME in source.attributes:
        raise InvalidArgument('Special timestamp attribute exists in wire message. Unable to parse message.')
    out.data = source.data
    for key, values in source.attributes.items():
        out.attributes[key] = _parse_attributes(values)
    if source.HasField('event_time'):
        out.attributes[PUBSUB_LITE_EVENT_TIME] = _encode_attribute_event_time_proto(source.event_time)
    return out