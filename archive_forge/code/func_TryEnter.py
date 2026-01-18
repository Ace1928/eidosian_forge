from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def TryEnter(self):
    """Trys to enter into the mode."""
    res = self.parser_.StartsWith(self.enter_seq_)
    if res:
        self.parser_.EmitBuffer()
        self.parser_.Advance(len(self.enter_seq_))
    return res