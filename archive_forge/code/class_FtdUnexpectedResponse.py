from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
class FtdUnexpectedResponse(Exception):
    """The exception to be raised in case of unexpected responses from 3d parties."""
    pass