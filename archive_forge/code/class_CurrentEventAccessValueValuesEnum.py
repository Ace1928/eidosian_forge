from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CurrentEventAccessValueValuesEnum(_messages.Enum):
    """Defines the level of data access when an event add-on is triggered.

    Values:
      UNSPECIFIED: Default value when nothing is set for eventAccess.
      METADATA: Gives event triggers the permission to access the metadata of
        events, such as event ID and calendar ID.
      READ: Gives event triggers access to all provided event fields including
        the metadata, attendees, and conference data.
      WRITE: Gives event triggers access to the metadata of events and the
        ability to perform all actions, including adding attendees and setting
        conference data.
      READ_WRITE: Gives event triggers access to all provided event fields
        including the metadata, attendees, and conference data and the ability
        to perform all actions.
    """
    UNSPECIFIED = 0
    METADATA = 1
    READ = 2
    WRITE = 3
    READ_WRITE = 4