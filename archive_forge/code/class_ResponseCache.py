import inspect
import logging
import os
import pickle
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._raylet as raylet
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.signature import extract_signature, get_signature
from ray._private.utils import check_oversized_function
from ray.util.client import ray
from ray.util.client.options import validate_options
class ResponseCache:
    """
    Cache for blocking method calls. Needed to prevent retried requests from
    being applied multiple times on the server, for example when the client
    disconnects. This is used to cache requests/responses sent through
    unary-unary RPCs to the RayletServicer.

    Note that no clean up logic is used, the last response for each thread
    will always be remembered, so at most the cache will hold N entries,
    where N is the number of threads on the client side. This relies on the
    assumption that a thread will not make a new blocking request until it has
    received a response for a previous one, at which point it's safe to
    overwrite the old response.

    The high level logic is:

    1. Before making a call, check the cache for the current thread.
    2. If present in the cache, check the request id of the cached
        response.
        a. If it matches the current request_id, then the request has been
            received before and we shouldn't re-attempt the logic. Wait for
            the response to become available in the cache, and then return it
        b. If it doesn't match, then this is a new request and we can
            proceed with calling the real stub. While the response is still
            being generated, temporarily keep (req_id, None) in the cache.
            Once the call is finished, update the cache entry with the
            new (req_id, response) pair. Notify other threads that may
            have been waiting for the response to be prepared.
    """

    def __init__(self):
        self.cv = threading.Condition()
        self.cache: Dict[int, Tuple[int, Any]] = {}

    def check_cache(self, thread_id: int, request_id: int) -> Optional[Any]:
        """
        Check the cache for a given thread, and see if the entry in the cache
        matches the current request_id. Returns None if the request_id has
        not been seen yet, otherwise returns the cached result.

        Throws an error if the placeholder in the cache doesn't match the
        request_id -- this means that a new request evicted the old value in
        the cache, and that the RPC for `request_id` is redundant and the
        result can be discarded, i.e.:

        1. Request A is sent (A1)
        2. Channel disconnects
        3. Request A is resent (A2)
        4. A1 is received
        5. A2 is received, waits for A1 to finish
        6. A1 finishes and is sent back to client
        7. Request B is sent
        8. Request B overwrites cache entry
        9. A2 wakes up extremely late, but cache is now invalid

        In practice this is VERY unlikely to happen, but the error can at
        least serve as a sanity check or catch invalid request id's.
        """
        with self.cv:
            if thread_id in self.cache:
                cached_request_id, cached_resp = self.cache[thread_id]
                if cached_request_id == request_id:
                    while cached_resp is None:
                        self.cv.wait()
                        cached_request_id, cached_resp = self.cache[thread_id]
                        if cached_request_id != request_id:
                            raise RuntimeError(f"Cached response doesn't match the id of the original request. This might happen if this request was received out of order. The result of the caller is no longer needed. ({request_id} != {cached_request_id})")
                    return cached_resp
                if not _id_is_newer(request_id, cached_request_id):
                    raise RuntimeError(f'Attempting to replace newer cache entry with older one. This might happen if this request was received out of order. The result of the caller is no longer needed. ({request_id} != {cached_request_id}')
            self.cache[thread_id] = (request_id, None)
        return None

    def update_cache(self, thread_id: int, request_id: int, response: Any) -> None:
        """
        Inserts `response` into the cache for `request_id`.
        """
        with self.cv:
            cached_request_id, cached_resp = self.cache[thread_id]
            if cached_request_id != request_id or cached_resp is not None:
                raise RuntimeError(f"Attempting to update the cache, but placeholder's do not match the current request_id. This might happen if this request was received out of order. The result of the caller is no longer needed. ({request_id} != {cached_request_id})")
            self.cache[thread_id] = (request_id, response)
            self.cv.notify_all()