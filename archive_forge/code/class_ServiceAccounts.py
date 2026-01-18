from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class ServiceAccounts(base.Group):
    """Create and manipulate service accounts.

     Create and manipulate IAM service accounts. A service account is a special
     Google account that belongs to your application or a VM, instead of to an
     individual end user. Your application uses the service account to call the
     Google API of a service, so that the users aren't directly involved.

     Note: Service accounts use client quotas for tracking usage.

     More information on service accounts can be found at:
     https://cloud.google.com/iam/docs/service-accounts
  """