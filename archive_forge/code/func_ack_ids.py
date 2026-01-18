from __future__ import absolute_import
import copy
import logging
import random
import threading
import time
import typing
from typing import Dict, Iterable, Optional, Union
from google.cloud.pubsub_v1.subscriber._protocol.dispatcher import _MAX_BATCH_LATENCY
from google.cloud.pubsub_v1.subscriber._protocol import requests
@property
def ack_ids(self) -> KeysView[str]:
    """The ack IDs of all leased messages."""
    return self._leased_messages.keys()