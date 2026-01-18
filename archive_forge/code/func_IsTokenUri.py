from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def IsTokenUri(uri):
    """Determine if the given URI is for requesting an access token."""
    if uri in TOKEN_URIS:
        return True
    metadata_regexp = '(metadata.google.internal|169.254.169.254)/computeMetadata/.*?/instance/service-accounts/.*?/token'
    impersonate_service_account = 'iamcredentials.googleapis.com/v.*?/projects/-/serviceAccounts/.*?:generateAccessToken'
    if re.search(metadata_regexp, uri) is not None:
        return True
    if re.search(impersonate_service_account, uri) is not None:
        return True
    return False