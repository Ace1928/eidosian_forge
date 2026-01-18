import hashlib
import random
from google.cloud.pubsublite.internal.wire.routing_policy import RoutingPolicy
from google.cloud.pubsublite.types.partition import Partition
from google.cloud.pubsublite_v1.types import PubSubMessage
Route the message using the key if set or round robin if unset.