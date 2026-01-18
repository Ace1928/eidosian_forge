import collections
from grpc.framework.interfaces.base import base
class _Completion(base.Completion, collections.namedtuple('_Completion', ('terminal_metadata', 'code', 'message'))):
    """A trivial implementation of base.Completion."""