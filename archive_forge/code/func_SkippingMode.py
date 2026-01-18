from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def SkippingMode(self, enter_seq, exit_seqs):
    return DDLParserMode(self, enter_seq, exit_seqs, None, True)