from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Entitlements(base.Group):
    """Manage Privileged Access Manager (PAM) Entitlements.

  The `gcloud pam entitlements` command group lets you manage Privileged Access
  Manager (PAM) Entitlements.

  ## EXAMPLES

  To create a new entitlement with the name `sample-entitlement`, under a
  project `sample-project`, location `global` and the entitlement body present
  in the `sample-entitlement.yaml` file, run:

      $ {command} create sample-entitlement --project=sample-project
      --location=global --entitlement-file=sample-entitlement.yaml

  To create a new entitlement with the name `sample-entitlement`, under a
  folder `sample-folder`, location `global` and the entitlement body present
  in the `sample-entitlement.yaml` file, run:

      $ {command} create sample-entitlement --folder=sample-folder
      --location=global --entitlement-file=sample-entitlement.yaml

  To create a new entitlement with the name `sample-entitlement`, under an
  organization `sample-organization`, location `global` and the entitlement
  body present in the `sample-entitlement.yaml` file, run:

      $ {command} create sample-entitlement --organization=sample-organization
      --location=global --entitlement-file=sample-entitlement.yaml

  To update an entitlement with the full name ``ENTITLEMENT_NAME'' and the new
  entitlement body present in the `sample-entitlement.yaml` file, run:

      $ {command} update ENTITLEMENT_NAME
      --entitlement-file=sample-entitlement.yaml

  To describe an entitlement with the full name ``ENTITLEMENT_NAME'', run:

      $ {command} describe ENTITLEMENT_NAME

  To search and list all entitlements under a project `sample-project` and
  location `global` for which you are a requester, run:

      $ {command} search --project=sample-project --location=global
      --caller-access-type=grant-requester

  To search and list all entitlements under a project `sample-project` and
  location `global` for which you are an approver, run:

      $ {command} search --project=sample-project --location=global
      --caller-access-type=grant-approver

  To search and list all entitlements under a folder `sample-folder` and
  location `global` for which you are a requester, run:

      $ {command} search --folder=sample-folder --location=global
      --caller-access-type=grant-requester

  To search and list all entitlements under a folder `sample-folder` and
  location `global` for which you are an approver, run:

      $ {command} search --folder=sample-folder --location=global
      --caller-access-type=grant-approver

  To search and list all entitlements under an organization
  `sample-organization` and location `global` for which you are a requester,
  run:

      $ {command} search --organization=sample-organization --location=global
      --caller-access-type=grant-requester

  To search and list all entitlements under an organization
  `sample-organization` and location `global` for which you are an approver,
  run:

      $ {command} search --organization=sample-organization --location=global
      --caller-access-type=grant-approver

  To list all entitlement under a project `sample-project` and location
  `global`, run:

      $ {command} list --project=sample-project --location=global

  To list all entitlement under a folder `sample-folder` and location
  `global`, run:

      $ {command} list --folder=sample-folder --location=global

  To list all entitlement under an organization `sample-organization` and
  location `global`, run:

      $ {command} list --organization=sample-organization --location=global

  To delete an entitlement with the full name ``ENTITLEMENT_NAME'' along with
  all grants under it, run:

      $ {command} delete ENTITLEMENT_NAME

  To export an entitlement with the full name ``ENTITLEMENT_NAME'' to a
  local YAML file `entitlement-file.yaml`, run:

      $ {command} export ENTITLEMENT_NAME --destination=entitlement-file.yaml

  """