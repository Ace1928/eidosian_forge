from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def _get_request_context(self):
    return oslo_context.get_current() or oslo_context.RequestContext()