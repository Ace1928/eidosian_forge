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
def ListImageVersionUpgrades(env_ref, release_track=base.ReleaseTrack.GA):
    """List of available image version upgrades for provided env_ref."""
    env_details = environments_api_util.Get(env_ref, release_track)
    proj_location_ref = env_ref.Parent()
    cur_image_version_id = env_details.config.softwareConfig.imageVersion
    cur_python_version = env_details.config.softwareConfig.pythonVersion
    return _BuildUpgradeCandidateList(proj_location_ref, cur_image_version_id, cur_python_version, release_track)