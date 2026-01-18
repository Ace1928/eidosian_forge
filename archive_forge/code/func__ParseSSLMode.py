from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ParseSSLMode(alloydb_messages, ssl_mode):
    if ssl_mode == 'ENCRYPTED_ONLY':
        return alloydb_messages.SslConfig.SslModeValueValuesEnum.ENCRYPTED_ONLY
    elif ssl_mode == 'ALLOW_UNENCRYPTED_AND_ENCRYPTED':
        return alloydb_messages.SslConfig.SslModeValueValuesEnum.ALLOW_UNENCRYPTED_AND_ENCRYPTED
    return None