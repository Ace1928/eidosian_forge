from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def LogMigrateSuccess(response, unused_ref):
    """Expected to contain displayName field, return "key" otherwise."""
    key_name = response.displayName or 'reCAPTCHA key to Enterprise'
    success_msg = 'Migration of {} succeeded.'.format(key_name)
    msg_len = len(success_msg)
    log.status.Print('-' * msg_len)
    log.status.Print(success_msg)
    log.status.Print('-' * msg_len)