from typing import List, Optional, Union
from ray._private.client_mode_hook import client_mode_hook
from ray._raylet import GcsClient
def _initialize_internal_kv(gcs_client: GcsClient):
    """Initialize the internal KV for use in other function calls."""
    global global_gcs_client, _initialized
    assert gcs_client is not None
    global_gcs_client = gcs_client
    _initialized = True