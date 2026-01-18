from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
class FtdConfigurationError(Exception):

    def __init__(self, msg, obj=None):
        super(FtdConfigurationError, self).__init__(msg)
        self.msg = msg
        self.obj = obj