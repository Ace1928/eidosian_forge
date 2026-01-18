from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
class AviCheckModeResponse(object):
    """
    Class to support ansible check mode.
    """

    def __init__(self, obj, status_code=200):
        self.obj = obj
        self.status_code = status_code

    def json(self):
        return self.obj