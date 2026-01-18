from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
class ComponentInfo(object):
    """Encapsulates information to be displayed for a component.

  Attributes:
    id: str, The component id.
    platform: str, The operating system and architecture of the platform.
    name: str, The display name of the component.
    current_version_string: str, The version of the component.
    is_hidden: bool, If the component is hidden.
    is_configuration: bool, True if this should be displayed in the packages
      section of the component manager.
    platform_required: bool, True if a platform-specific executable is
      required.
  """

    def __init__(self, component_id, snapshot, platform_filter=None):
        """Create a new component info container.

    Args:
      component_id: str, The id of this component.
      snapshot: ComponentSnapshot, The snapshot from which to create info from.
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.
    """
        self._id = component_id
        self._snapshot = snapshot
        self._component = snapshot.ComponentFromId(component_id)
        self._platform_filter = platform_filter

    @property
    def id(self):
        return self._id

    @property
    def platform(self):
        return self._platform_filter

    @property
    def current_version_string(self):
        return self._component.version.version_string

    @property
    def name(self):
        return self._component.details.display_name

    @property
    def is_hidden(self):
        return self._component.is_hidden

    @property
    def is_configuration(self):
        return self._component.is_configuration

    @property
    def platform_required(self):
        return self._component.platform_required

    @property
    def size(self):
        return self._snapshot.GetEffectiveComponentSize(self._id, platform_filter=self._platform_filter)

    def __str__(self):
        return '{name} ({id})\t[{current_version}]'.format(name=self.name, id=self.id, current_version=self.current_version_string)