import itertools
from oslo_serialization import jsonutils
import webob
class _AuthTokenResponse(webob.Response):
    default_content_type = None