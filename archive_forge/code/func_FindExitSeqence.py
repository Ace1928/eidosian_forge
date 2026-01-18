from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def FindExitSeqence(self):
    """Finds a matching exit sequence."""
    for s in self.exit_seqs_:
        if self.parser_.StartsWith(s):
            return s
    return None