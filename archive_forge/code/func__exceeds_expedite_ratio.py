from typing import List, Optional
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
def _exceeds_expedite_ratio(pending: int, client: int):
    if client <= 0:
        return False
    return pending / client >= _EXPEDITE_BATCH_REQUEST_RATIO