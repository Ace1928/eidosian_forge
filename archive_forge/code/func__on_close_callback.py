from __future__ import absolute_import
import typing
from typing import Any
from typing import Union
from google.cloud.pubsub_v1 import futures
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
def _on_close_callback(self, manager: 'StreamingPullManager', result: Any):
    if self.done():
        return
    if result is None:
        self.set_result(True)
    else:
        self.set_exception(result)