from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
@base.Hidden
class Rules(base.Group):
    """Manage Artifact Registry rules.

  ## EXAMPLES
  To list all rules in the current project and `artifacts/repository` and
  `artifacts/location` properties are set,
  run:

      $ {command} list

  To list rules under repository my-repo in the current project and location,
  run:

      $ {command} list --repository=my-repo

  To delete rule `my-rule` under repository my-repo in the current project and
  location, run:

      $ {command} delete my-rule --repository=my-repo
  """
    category = base.CI_CD_CATEGORY