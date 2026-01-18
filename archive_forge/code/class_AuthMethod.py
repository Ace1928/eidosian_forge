from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
class AuthMethod(enum.Enum):
    AWS_SIGNATURE_V2 = 'AWS_SIGNATURE_V2'
    AWS_SIGNATURE_V4 = 'AWS_SIGNATURE_V4'