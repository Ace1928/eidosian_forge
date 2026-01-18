from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def _CanonicalizeValue(value):
    """Canonicalizes secret value reference to the secret version resource name.

  Output format: `projects/{project}/secrets/{secret}/versions/{version}`.
  The project in the above reference will be * if the user used a default
  project secret.

  Args:
    value: Secret value reference as a string.

  Returns:
    Canonicalized secret value reference.
  """
    dp_secret_ref_match = _DEFAULT_PROJECT_SECRET_REF_PATTERN.search(value)
    secret_version_res_ref_match = _SECRET_VERSION_RESOURCE_REF_PATTERN.search(value)
    secret_version_ref_match = _SECRET_VERSION_REF_PATTERN.search(value)
    if dp_secret_ref_match:
        return 'projects/{project}/secrets/{secret}/versions/{version}'.format(project=_DEFAULT_PROJECT_IDENTIFIER, secret=dp_secret_ref_match.group('secret'), version=dp_secret_ref_match.group('version'))
    elif secret_version_res_ref_match:
        return value
    elif secret_version_ref_match:
        return 'projects/{project}/secrets/{secret}/versions/{version}'.format(project=secret_version_ref_match.group('project'), secret=secret_version_ref_match.group('secret'), version=secret_version_ref_match.group('version'))
    raise ArgumentTypeError("Secrets value configuration must match the pattern 'SECRET:VERSION' or 'projects/{{PROJECT}}/secrets/{{SECRET}}:{{VERSION}}' or 'projects/{{PROJECT}}/secrets/{{SECRET}}/versions/{{VERSION}}' where VERSION is a number or the label 'latest' [{}]".format(value))