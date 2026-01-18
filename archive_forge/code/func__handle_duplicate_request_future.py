from __future__ import absolute_import
from __future__ import division
import functools
import itertools
import logging
import math
import time
import threading
import typing
from typing import List, Optional, Sequence, Union
import warnings
from google.api_core.retry import exponential_sleep_generator
from google.cloud.pubsub_v1.subscriber._protocol import helper_threads
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
def _handle_duplicate_request_future(self, exactly_once_delivery_enabled: bool, item: Union[requests.AckRequest, requests.ModAckRequest, requests.NackRequest]) -> None:
    _LOGGER.debug('This is a duplicate %s with the same ack_id: %s.', type(item), item.ack_id)
    if item.future:
        if exactly_once_delivery_enabled:
            item.future.set_exception(ValueError(f'Duplicate ack_id for {type(item)}'))
        else:
            item.future.set_result(AcknowledgeStatus.SUCCESS)