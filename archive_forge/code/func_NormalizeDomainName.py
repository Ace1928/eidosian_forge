from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.domains import operations
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def NormalizeDomainName(domain):
    """Normalizes domain name (including punycoding)."""
    if not domain:
        raise exceptions.Error('Empty domain name')
    try:
        normalized = domain.encode('idna').decode()
        normalized = normalized.lower().rstrip('.')
    except UnicodeError as e:
        raise exceptions.Error("Invalid domain name '{}': {}.".format(domain, e))
    return normalized