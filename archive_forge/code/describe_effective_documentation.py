from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc.manage.etd import clients
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import flags
from googlecloudsdk.command_lib.scc.manage import parsing
Get the effective details of a Event Threat Detection effective custom module.

  Get the effective details of a Event Threat Detection effective custom module.
  It retrieves a custom module with its effective enablement state.

  ## EXAMPLES

  To get the effective details of a Event Threat Detection custom module with ID
  `123456` for organization `123`, run:

    $ {command} 123456 --organization=123

  To get the effective details of a Event Threat Detection custom module with ID
  `123456` for folder `456`, run:

    $ {command} 123456 --folder=456

  To get the effective details of a Event Threat Detection custom module with ID
  `123456` for project `789`, run:

    $ {command} 123456 --project=789

  You can also specify the parent more generally:

    $ {command} 123456 --parent=organizations/123

  Or just specify the fully qualified module name:

    $ {command}
    organizations/123/locations/global/effectiveEventThreatDetectionCustomModules/123456
  