from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def ParsePublishingOptions(args):
    """Parses the PublshingOptions proto message from the args."""
    messages = privateca_base.GetMessagesModule('v1')
    publish_ca_cert = args.publish_ca_cert
    publish_crl = args.publish_crl
    encoding_format = ParseEncodingFormatFlag(args)
    is_devops_tier = args.IsKnownAndSpecified('tier') and ParseTierFlag(args) == messages.CaPool.TierValueValuesEnum.DEVOPS
    if is_devops_tier:
        if args.IsSpecified('publish_crl') and publish_crl:
            raise exceptions.InvalidArgumentException('--publish-crl', 'CRL publication is not supported in the DevOps tier.')
        publish_crl = False
    return messages.PublishingOptions(publishCaCert=publish_ca_cert, publishCrl=publish_crl, encodingFormat=encoding_format)