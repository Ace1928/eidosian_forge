import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
def _normalize_grpc_url(self, address):
    return common.GRPC_URL_PREFIX + address if not address.startswith(common.GRPC_URL_PREFIX) else address