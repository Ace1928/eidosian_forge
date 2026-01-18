import sys
import traceback
from oslo_config import cfg
from oslo_utils import reflection
import webob
from heat.common import exception
from heat.common import serializers
from heat.common import wsgi
def _map_exception_to_error(self, class_exception):
    if class_exception == Exception:
        return webob.exc.HTTPInternalServerError
    if class_exception.__name__ not in self.error_map:
        return self._map_exception_to_error(class_exception.__base__)
    return self.error_map[class_exception.__name__]