from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _GetEntriesClass(self, current_acl):
    for acl_entry in current_acl:
        return acl_entry.__class__
    return apitools_messages.ObjectAccessControl().__class__