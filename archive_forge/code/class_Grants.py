from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Grants(base.Group):
    """Manage Privileged Access Manager (PAM) Grants.

  The `gcloud pam grants` command group lets you manage Privileged
  Access Manager (PAM) Grants.

  ## EXAMPLES

  To create a new grant under an entitlement with the full name
  ``ENTITLEMENT_NAME'', with requested duration `1 hour 30 minutes`, a
  justification `some justification` and two additional email recipients
  `abc@example.com` and `xyz@example.com`, run:

      $ {command} create --entitlement=ENTITLEMENT_NAME
      --requested-duration=5400s
      --justification="some justification"
      --additional-email-recipients=abc@example.com,xyz@example.com

  To describe a grant with the full name ``GRANT_NAME'', run:

      $ {command} describe GRANT_NAME

  To list all grants under an entitlement with the full name
  ``ENTITLEMENT_NAME'', run:

      $ {command} list --entitlement=ENTITLEMENT_NAME

  To deny a grant with the full name ``GRANT_NAME'' and a reason
  `denial reason`, run:

      $ {command} deny GRANT_NAME --reason="denial reason"

  To approve a grant with the full name ``GRANT_NAME'' and a reason
  `approval reason`, run:

      $ {command} approve GRANT_NAME --reason="approval reason"

  To revoke a grant with the full name ``GRANT_NAME'' and a reason
  `revoke reason`, run:

      $ {command} revoke GRANT_NAME --reason="revoke reason"

  To search and list all grants under an entitlement with the full name
  ``ENTITLEMENT_NAME'', which you had created, run:

      $ {command} search --entitlement=ENTITLEMENT_NAME
      --caller-relationship=had-created

  To search and list all grants under an entitlement with the full name
  ``ENTITLEMENT_NAME'', which you had approved or denied, run:

      $ {command} search --entitlement=ENTITLEMENT_NAME
      --caller-relationship=had-approved

  To search and list all grants under an entitlement with the full name
  ``ENTITLEMENT_NAME'', which you can approve, run:

      $ {command} search --entitlement=ENTITLEMENT_NAME
      --caller-relationship=can-approve

  """