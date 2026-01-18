import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
Create or update the given endpoint.

        This method is idempotent - if the endpoint already exists it will be
        updated to match the given parameters. Calling this twice with the same
        arguments is a no-op.
        