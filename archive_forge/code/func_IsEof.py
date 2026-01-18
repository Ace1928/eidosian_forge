from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def IsEof(self):
    return self.next_index_ == len(self.ddl_)