import asyncio
from typing import Set
from google.cloud.pubsublite.internal.wire.assigner import Assigner
from google.cloud.pubsublite.types.partition import Partition
Only returns an assignment the first iteration.