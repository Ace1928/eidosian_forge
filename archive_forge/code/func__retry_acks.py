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
def _retry_acks(self, requests_to_retry):
    retry_delay_gen = exponential_sleep_generator(initial=_MIN_EXACTLY_ONCE_DELIVERY_ACK_MODACK_RETRY_DURATION_SECS, maximum=_MAX_EXACTLY_ONCE_DELIVERY_ACK_MODACK_RETRY_DURATION_SECS)
    while requests_to_retry:
        time_to_wait = next(retry_delay_gen)
        _LOGGER.debug('Retrying {len(requests_to_retry)} ack(s) after delay of ' + str(time_to_wait) + ' seconds')
        time.sleep(time_to_wait)
        ack_reqs_dict = {req.ack_id: req for req in requests_to_retry}
        requests_completed, requests_to_retry = self._manager.send_unary_ack(ack_ids=[req.ack_id for req in requests_to_retry], ack_reqs_dict=ack_reqs_dict)
        assert len(requests_to_retry) <= _ACK_IDS_BATCH_SIZE, 'Too many requests to be retried.'
        self.drop(requests_completed)