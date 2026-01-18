import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import json_format
from google.rpc import code_pb2
def _refresh_and_update(self, retry=None):
    """Refresh the operation and update the result if needed.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the RPC.
        """
    if not self._operation.done:
        self._operation = self._refresh(retry=retry) if retry else self._refresh()
        self._set_result_from_operation()