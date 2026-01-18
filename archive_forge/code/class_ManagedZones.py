from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class ManagedZones(base.Group):
    """Manage your Cloud DNS managed-zones.

  Manage your Cloud DNS managed-zones. See
  [Managing Zones](https://cloud.google.com/dns/zones/) for details.

  ## EXAMPLES

  To create a managed-zone, run:

    $ {command} create my-zone --description="My Zone" --dns-name="my.zone.com."

  To delete a managed-zone, run:

    $ {command} delete my-zone

  To view the details of a managed-zone, run:

    $ {command} describe my-zone

  To see the list of all managed-zones, run:

    $ {command} list
  """
    pass