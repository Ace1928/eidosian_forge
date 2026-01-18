import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def _build_middleware(self):
    return context.ContextMiddleware(None)