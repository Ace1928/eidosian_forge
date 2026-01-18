import collections
import datetime
import logging
import os
import threading
from typing import (
import grpc
from grpc.experimental import experimental_api
@staticmethod
def _perform_evictions():
    while True:
        with ChannelCache._lock:
            ChannelCache._eviction_ready.set()
            if not ChannelCache._singleton._mapping:
                ChannelCache._condition.wait()
            elif len(ChannelCache._singleton._mapping) > _MAXIMUM_CHANNELS:
                key = next(iter(ChannelCache._singleton._mapping.keys()))
                ChannelCache._singleton._evict_locked(key)
            else:
                key, (_, eviction_time) = next(iter(ChannelCache._singleton._mapping.items()))
                now = datetime.datetime.now()
                if eviction_time <= now:
                    ChannelCache._singleton._evict_locked(key)
                    continue
                else:
                    time_to_eviction = (eviction_time - now).total_seconds()
                    ChannelCache._condition.wait(timeout=time_to_eviction)