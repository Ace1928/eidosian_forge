from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def _get_matching_grant_identifier_to_remove_for_shim(existing_grant, grant_identifiers):
    """Shim-only support for case-insensitive matching on non-entity metadata.

  Ports the logic here:
  https://github.com/GoogleCloudPlatform/gsutil/blob/0d9d0175b2b10430471c7b744646e56210f991d3/gslib/utils/acl_helper.py#L291

  Args:
    existing_grant (BucketAccessControl|ObjectAccessControl): A grant currently
      in a resource's access control list.
    grant_identifiers (Iterable[str]): User input specifying the grants to
      remove.

  Returns:
    A string matching existing_grant in grant_identifiers if one exists.
      Otherwise, None. Note that this involves preserving the original case of
      the identifier, despite the fact that this function performs a
      case-insensitive comparison.
  """
    normalized_identifier_to_original = {identifier.lower(): identifier for identifier in grant_identifiers}
    if existing_grant.entityId:
        normalized_entity_id = existing_grant.entityId.lower()
        if normalized_entity_id in normalized_identifier_to_original:
            return normalized_identifier_to_original[normalized_entity_id]
    if existing_grant.email:
        normalized_email = existing_grant.email.lower()
        if normalized_email in normalized_identifier_to_original:
            return normalized_identifier_to_original[normalized_email]
    if existing_grant.domain:
        normalized_domain = existing_grant.domain.lower()
        if normalized_domain in normalized_identifier_to_original:
            return normalized_identifier_to_original[normalized_domain]
    if existing_grant.projectTeam:
        normalized_identifier = '{}-{}'.format(existing_grant.projectTeam.team, existing_grant.projectTeam.projectNumber).lower()
        if normalized_identifier in normalized_identifier_to_original:
            return normalized_identifier_to_original[normalized_identifier]
    if existing_grant.entity:
        normalized_entity = existing_grant.entity.lower()
        if normalized_entity in normalized_identifier_to_original and normalized_entity in ['allusers', 'allauthenticatedusers']:
            return normalized_identifier_to_original[normalized_entity]