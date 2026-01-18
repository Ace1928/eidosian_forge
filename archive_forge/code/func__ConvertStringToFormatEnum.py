from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def _ConvertStringToFormatEnum(type_, messages):
    """Convert string values to Enum object type."""
    if type_ == flags.KeyTypes.RS256.choice_name or type_ == flags.KeyTypes.RSA_X509_PEM.choice_name:
        return messages.PublicKeyCredential.FormatValueValuesEnum.RSA_X509_PEM
    elif type_ == flags.KeyTypes.RSA_PEM.choice_name:
        return messages.PublicKeyCredential.FormatValueValuesEnum.RSA_PEM
    elif type_ == flags.KeyTypes.ES256_X509_PEM.choice_name:
        return messages.PublicKeyCredential.FormatValueValuesEnum.ES256_X509_PEM
    elif type_ == flags.KeyTypes.ES256.choice_name or type_ == flags.KeyTypes.ES256_PEM.choice_name:
        return messages.PublicKeyCredential.FormatValueValuesEnum.ES256_PEM
    else:
        raise ValueError('Invalid key type [{}]'.format(type_))