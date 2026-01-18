from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _ValidateCertificateFormat(self, args, field):
    if not hasattr(args, field) or not args.IsSpecified(field):
        return True
    certificate = getattr(args, field)
    cert = certificate.strip()
    cert_lines = cert.split('\n')
    if not cert_lines[0].startswith('-----') or not cert_lines[-1].startswith('-----'):
        raise calliope_exceptions.InvalidArgumentException(field, 'The certificate does not appear to be in PEM format:\n{0}'.format(cert))