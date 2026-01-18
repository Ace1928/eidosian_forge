from inspect import getfullargspec
from oslo_config import cfg
import webob.dec
import webob.request
import webob.response
class NoContentTypeRequest(webob.request.Request):
    ResponseClass = NoContentTypeResponse