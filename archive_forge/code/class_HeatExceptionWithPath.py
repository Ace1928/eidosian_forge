import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class HeatExceptionWithPath(HeatException):
    msg_fmt = _('%(error)s%(path)s%(message)s')

    def __init__(self, error=None, path=None, message=None):
        self.error = error or ''
        self.path = []
        if path is not None:
            if isinstance(path, list):
                self.path = path
            elif isinstance(path, str):
                self.path = [path]
        result_path = ''
        for path_item in self.path:
            if isinstance(path_item, int) or path_item.isdigit():
                result_path += '[%s]' % path_item
            elif len(result_path) > 0:
                result_path += '.%s' % path_item
            else:
                result_path = path_item
        self.error_message = message or ''
        super(HeatExceptionWithPath, self).__init__(error='%s: ' % self.error if self.error != '' else '', path='%s: ' % result_path if len(result_path) > 0 else '', message=self.error_message)