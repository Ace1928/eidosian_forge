import datetime
import uuid
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def GenerateRequestId():
    return str(uuid.uuid4())