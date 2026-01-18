from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from typing import List
from googlecloudsdk.api_lib.dataplex import util as dataplex_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
def IsoDateTime(datetime_str: str) -> str:
    """Parses datetime string, validates it and outputs the new datetime string in ISO format."""
    return arg_parsers.Datetime.Parse(datetime_str).isoformat()