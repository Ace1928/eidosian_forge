from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def StartsWith(self, s):
    return self.ddl_[self.next_index_:].startswith(s)