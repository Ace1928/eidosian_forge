from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def _IsZeroOrPositiveNumber(string):
    if string.isdigit():
        return int(string) >= 0
    return False