import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class StackValidationFailed(HeatExceptionWithPath):

    def __init__(self, error=None, path=None, message=None, resource=None):
        if path is None:
            path = []
        elif isinstance(path, str):
            path = [path]
        if resource is not None and (not path):
            path = [resource.stack.t.get_section_name(resource.stack.t.RESOURCES), resource.name]
        if isinstance(error, Exception):
            if isinstance(error, StackValidationFailed):
                str_error = error.error
                message = error.error_message
                path = path + error.path
                self.args = error.args
            else:
                str_error = str(type(error).__name__)
                message = str(error)
        else:
            str_error = error
        super(StackValidationFailed, self).__init__(error=str_error, path=path, message=message)