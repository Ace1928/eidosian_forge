from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def _ExtractApiNameFromDomain(domain):
    return domain.split('.')[0]