import typing
from typing import Optional
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher._sequencer import base
from google.pubsub_v1 import types as gapic_types
def _set_batch(self, batch: '_batch.thread.Batch') -> None:
    self._current_batch = batch