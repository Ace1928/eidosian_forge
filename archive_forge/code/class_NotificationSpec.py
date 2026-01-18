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
class NotificationSpec(object):
    """Data type for a update notification object.

  Attributes:
    condition: Condition, The settings for whether or not this notification
      should be activated by a particular installation.
    trigger: Trigger, The settings for whether to trigger an activated
      notification on a particular command execution.
    notification: Notification, The settings about how to actually express the
      notification to the user once it is triggered.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        """Converts a dictionary object to an instantiated NotificationSpec class.

    Args:
      dictionary: The Dictionary to to convert from.

    Returns:
      A NotificationSpec object initialized from the dictionary object.
    """
        p = DictionaryParser(cls, dictionary)
        p.Parse('id', required=True)
        p.Parse('condition', default={}, func=Condition.FromDictionary)
        p.Parse('trigger', default={}, func=Trigger.FromDictionary)
        p.Parse('notification', default={}, func=Notification.FromDictionary)
        return cls(**p.Args())

    def ToDictionary(self):
        """Converts a Component object to a Dictionary object.

    Returns:
      A Dictionary object initialized from self.
    """
        w = DictionaryWriter(self)
        w.Write('id')
        w.Write('condition', func=Condition.ToDictionary)
        w.Write('trigger', func=Trigger.ToDictionary)
        w.Write('notification', func=Notification.ToDictionary)
        return w.Dictionary()

    def __init__(self, id, condition, trigger, notification):
        self.id = id
        self.condition = condition
        self.trigger = trigger
        self.notification = notification