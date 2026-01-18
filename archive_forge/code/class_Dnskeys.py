from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Dnskeys(base.Group):
    """Manage Cloud DNS DNSKEY records.

  The commands in this group manage Cloud DNS DNS key resources. A DNS key
  resource represents a cryptographic signing key for use in DNSSEC; a DNSKEY
  record contains a public key used for digitally signing zone data.

  For more information, including instructions for managing and using DNS keys,
  see the [documentation for DNSSEC](https://cloud.google.com/dns/dnssec).
  """
    pass