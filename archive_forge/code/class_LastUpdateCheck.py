from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
class LastUpdateCheck(object):
    """Top level object for the cache of the last time an update check was done.

  Attributes:
    version: int, The schema version number.  A different number is considered
      incompatible.
    no_update: bool, True if this installation should not attempted to be
      updated.
    message: str, A message to display to the user if they are updating to this
      new schema version.
    url: str, The URL to grab a fresh Cloud SDK bundle.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        p = DictionaryParser(cls, dictionary)
        p.Parse('last_update_check_time', default=0)
        p.Parse('last_update_check_revision', default=0)
        p.ParseList('notifications', default=[], func=NotificationSpec.FromDictionary)
        p.ParseDict('last_nag_times', default={})
        return cls(**p.Args())

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.Write('last_update_check_time')
        w.Write('last_update_check_revision')
        w.WriteList('notifications', func=NotificationSpec.ToDictionary)
        w.WriteDict('last_nag_times')
        return w.Dictionary()

    def __init__(self, last_update_check_time, last_update_check_revision, notifications, last_nag_times):
        self.last_update_check_time = last_update_check_time
        self.last_update_check_revision = last_update_check_revision
        self.notifications = notifications
        self.last_nag_times = last_nag_times