from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def PreservingMode(self, enter_seq, exit_seqs, escape_sequences):
    return DDLParserMode(self, enter_seq, exit_seqs, escape_sequences, False)