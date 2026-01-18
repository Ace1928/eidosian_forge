import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class ConvertedException(webob.exc.WSGIHTTPException):

    def __init__(self, code=0, title='', explanation=''):
        self.code = code
        self.title = title
        self.explanation = explanation
        super(ConvertedException, self).__init__()