import collections
import sys
from google.rpc import status_pb2
import grpc
from ._common import GRPC_DETAILS_METADATA_KEY
from ._common import code_to_grpc_status_code
class _Status(collections.namedtuple('_Status', ('code', 'details', 'trailing_metadata')), grpc.Status):
    pass