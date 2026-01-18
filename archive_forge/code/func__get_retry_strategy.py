from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def _get_retry_strategy():
    retry_strategy_builder = RetryStrategyBuilder(max_attempts_check=True, max_attempts=10, retry_max_wait_between_calls_seconds=30, retry_base_sleep_time_seconds=3, backoff_type=oci.retry.BACKOFF_FULL_JITTER_EQUAL_ON_THROTTLE_VALUE)
    retry_strategy_builder.add_service_error_check(service_error_retry_config={429: [], 400: ['QuotaExceeded', 'LimitExceeded'], 409: ['Conflict']}, service_error_retry_on_any_5xx=True)
    return retry_strategy_builder.get_retry_strategy()