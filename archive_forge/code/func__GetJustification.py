from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def _GetJustification(email, justification):
    if email is not None and justification is not None:
        return 'email: %s. %s' % (email, justification)
    if email is None:
        return justification
    if justification is None:
        return 'email: %s.' % email
    return None