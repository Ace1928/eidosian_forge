from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _ThrowError(msg):
    raise CommandException('{0} is not a valid ACL change\n{1}'.format(self.raw_descriptor, msg))