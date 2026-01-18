from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def ExitMode(self):
    self.logger_.debug('exit mode: %s at index: %d', self.mode_.enter_seq_, self.next_index_)
    self.mode_ = None