from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import dns_util
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
Configure DNS settings of a Cloud Domains registration.

  Configure DNS settings of a Cloud Domains registration.

  In most cases, this command is used for changing the authoritative name
  servers and DNSSEC options for the given domain. However, advanced options
  like glue records are available.

  When using Cloud DNS Zone or Google Domains name servers, DNSSEC will be
  enabled by default where possible. You can disable it using the
  --disable-dnssec flag.

  ## EXAMPLES

  To start an interactive flow to configure DNS settings for ``example.com'',
  run:

    $ {command} example.com

  To use Cloud DNS managed-zone ``example-zone'' for ``example.com'', run:

    $ {command} example.com --cloud-dns-zone=example-zone

  If the managed-zone is signed, DNSSEC will be enabled for the domain.

  To change DNS settings for ``example.com'' according to information from a
  YAML file ``dns_settings.yaml'', run:

    $ {command} example.com --dns-settings-from-file=dns_settings.yaml

  To disable DNSSEC, run:

    $ {command} example.com --disable-dnssec

  