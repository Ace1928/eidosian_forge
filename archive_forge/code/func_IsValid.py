import datetime
import uuid
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def IsValid(self, s):
    if not s:
        return False
    return s[:self.key_len].upper() in self.options.keys()