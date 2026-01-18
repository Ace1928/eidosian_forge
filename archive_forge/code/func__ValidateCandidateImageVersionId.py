from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import image_versions_util as image_version_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core.util import semver
def _ValidateCandidateImageVersionId(current_image_version_id, candidate_image_version_id):
    """Determines if candidate version is a valid upgrade from current version.

  Args:
    current_image_version_id: current image version
    candidate_image_version_id: image version requested for upgrade

  Returns:
    UpgradeValidator namedtuple containing True and None error message if
    given version upgrade between given versions is valid, otherwise False and
    error message with problems description.
  """
    upgrade_validator = UpgradeValidator(True, None)
    if current_image_version_id == candidate_image_version_id:
        error_message = 'Existing and requested image versions are equal ({}). Select image version newer than current to perform upgrade.'.format(current_image_version_id)
        upgrade_validator = UpgradeValidator(False, error_message)
    parsed_curr = _ImageVersionItem(image_ver=current_image_version_id)
    parsed_cand = _ImageVersionItem(image_ver=candidate_image_version_id)
    has_alias_or_major_only_composer_ver = parsed_cand.composer_contains_alias or parsed_curr.composer_contains_alias
    if has_alias_or_major_only_composer_ver:
        upgrade_validator = _IsComposerMajorOnlyVersionUpgradeCompatible(parsed_curr, parsed_cand)
    elif upgrade_validator.upgrade_valid:
        upgrade_validator = _IsVersionUpgradeCompatible(parsed_curr.composer_ver, parsed_cand.composer_ver, 'Composer')
    if upgrade_validator.upgrade_valid and (not parsed_cand.airflow_contains_alias):
        upgrade_validator = _IsVersionUpgradeCompatible(parsed_curr.airflow_ver, parsed_cand.airflow_ver, 'Airflow')
    return upgrade_validator