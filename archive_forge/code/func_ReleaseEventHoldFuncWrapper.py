from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def ReleaseEventHoldFuncWrapper(cls, name_expansion_result, thread_state=None):
    log_template = 'Releasing Event-Based Hold on %s...'
    object_metadata_update = apitools_messages.Object(eventBasedHold=False)
    cls.ObjectUpdateMetadataFunc(object_metadata_update, log_template, name_expansion_result, thread_state=thread_state)