from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def GetIamCredentialsClientAndMessages():
    client = apis.GetClientInstance('iamcredentials', 'v1')
    return (client, client.MESSAGES_MODULE)