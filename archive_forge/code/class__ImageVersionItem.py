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
class _ImageVersionItem(object):
    """Class used to dissect and analyze image version components and strings."""

    def __init__(self, image_ver=None, composer_ver=None, airflow_ver=None):
        image_version_regex = '^composer-(\\d+(?:(?:\\.\\d+\\.\\d+(?:-[a-z]+\\.\\d+)?)?)?|latest)-airflow-(\\d+(?:\\.\\d+(?:\\.\\d+)?)?)'
        composer_version_alias_regex = '^(\\d+|latest)$'
        airflow_version_alias_regex = '^(\\d+|\\d+\\.\\d+)$'
        if image_ver is not None:
            iv_parts = re.findall(image_version_regex, image_ver)[0]
            self.composer_ver = iv_parts[0]
            self.airflow_ver = iv_parts[1]
        if composer_ver is not None:
            self.composer_ver = composer_ver
        if airflow_ver is not None:
            self.airflow_ver = airflow_ver
        self.composer_contains_alias = re.match(composer_version_alias_regex, self.composer_ver)
        self.airflow_contains_alias = re.match(airflow_version_alias_regex, self.airflow_ver)

    def GetImageVersionString(self):
        return 'composer-{}-airflow-{}'.format(self.composer_ver, self.airflow_ver)