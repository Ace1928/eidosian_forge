from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class GitCredentialHelperError(exceptions.Error):
    """Raised for issues related to passing auth credentials to Git."""