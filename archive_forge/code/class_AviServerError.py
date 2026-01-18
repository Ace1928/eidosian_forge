from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
class AviServerError(APIError):

    def __init__(self, arg, rsp=None):
        super(AviServerError, self).__init__(arg, rsp)