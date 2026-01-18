from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSelfManagedCertificateDataFlagsToParser(parser, is_required):
    """Adds certificate file and private key file flags."""
    cert_flag = base.Argument('--certificate-file', help='The certificate data in PEM-encoded form.', type=arg_parsers.FileContents(), required=True)
    key_flag = base.Argument('--private-key-file', help='The private key data in PEM-encoded form.', type=arg_parsers.FileContents(), required=True)
    group = base.ArgumentGroup(help='Arguments to configure self-managed certificate data.', required=is_required, category=base.COMMONLY_USED_FLAGS if not is_required else None)
    group.AddArgument(cert_flag)
    group.AddArgument(key_flag)
    group.AddToParser(parser)