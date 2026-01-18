from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
@enum.unique
class ReleaseTrack(enum.Enum):
    """An enum representing the release track of a command or command group.

  The release track controls where a command appears.  The default of GA means
  it will show up under gcloud.  If you enable a command or group for the alpha,
  beta, or preview tracks, those commands will be duplicated under those groups
  as well.
  """

    def __init__(self, prefix, help_tag, help_note):
        self.prefix = prefix
        self.help_tag = help_tag
        self.help_note = help_note

    def __str__(self):
        return self.name

    @property
    def id(self):
        return self.name
    GA = (None, None, None)
    BETA = ('beta', '{0}(BETA){0} '.format(MARKDOWN_BOLD), 'This command is currently in beta and might change without notice.')
    ALPHA = ('alpha', '{0}(ALPHA){0} '.format(MARKDOWN_BOLD), 'This command is currently in alpha and might change without notice. If this command fails with API permission errors despite specifying the correct project, you might be trying to access an API with an invitation-only early access allowlist.')

    @staticmethod
    def AllValues():
        """Gets all possible enum values.

    Returns:
      list, All the enum values.
    """
        return list(ReleaseTrack)

    @staticmethod
    def FromPrefix(prefix):
        """Gets a ReleaseTrack from the given release track prefix.

    Args:
      prefix: str, The prefix string that might be a release track name.

    Returns:
      ReleaseTrack, The corresponding object or None if the prefix was not a
      valid release track.
    """
        for track in ReleaseTrack:
            if track.prefix == prefix:
                return track
        return None

    @staticmethod
    def FromId(id):
        """Gets a ReleaseTrack from the given release track prefix.

    Args:
      id: str, The id string that must be a release track name.

    Raises:
      ValueError: For unknown release track ids.

    Returns:
      ReleaseTrack, The corresponding object.
    """
        try:
            return ReleaseTrack[id]
        except KeyError:
            raise ValueError('Unknown release track id [{}].'.format(id))