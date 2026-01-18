from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class LevelCondition(base.Group):
    """Manage Access Context Manager level conditions.

  An access level is a classification of requests based on raw attributes of
  that request (e.g. IP address, device identity, time of day, etc.). These
  individual attributes are called conditions.
  """