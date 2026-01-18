from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.secrets import api as secrets_api
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.secrets import args as secrets_args
from googlecloudsdk.command_lib.secrets import fmt as secrets_fmt
from googlecloudsdk.command_lib.secrets import util as secrets_util
from googlecloudsdk.command_lib.util import crc32c
Access a secret version's data.

  Access the data for the specified secret version.

  ## EXAMPLES

  Access the data for version 123 of the secret 'my-secret':

    $ {command} 123 --secret=my-secret

  Note: The output will be formatted as UTF-8 which can corrupt binary secrets.

  To write raw bytes to a file use --out-file flag:

    $ {command} 123 --secret=my-secret --out-file=/tmp/secret

  To get the raw bytes, have Google Cloud CLI print the response as
  base64-encoded and decode:

    $ {command} 123 --secret=my-secret --format='get(payload.data)' | tr '_-' '/+' | base64 -d
  