from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def UpdateObjectMetadataExceptionHandler(cls, e):
    """Exception handler that maintains state about post-completion status."""
    cls.logger.error(e)
    cls.everything_set_okay = False