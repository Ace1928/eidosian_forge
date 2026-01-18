import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
class SpecialTZInfo(datetime.tzinfo):

    def __init__(self, offset):
        super(SpecialTZInfo, self).__init__()
        self.offset = offset

    def __repr__(self):
        s = 'TimeZoneOffset(' + repr(self.offset) + ')'
        if not kwargs.get('no_modules'):
            s = 'apitools.base.protorpclite.util.' + s
        return s