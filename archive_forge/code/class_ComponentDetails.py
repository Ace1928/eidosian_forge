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
class ComponentDetails(object):
    """Encapsulates some general information about the component.

  Attributes:
    display_name: str, The user facing name of the component.
    description: str, A little more details about what the component does.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        p = DictionaryParser(cls, dictionary)
        p.Parse('display_name', required=True)
        p.Parse('description', required=True)
        return cls(**p.Args())

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.Write('display_name')
        w.Write('description')
        return w.Dictionary()

    def __init__(self, display_name, description):
        self.display_name = display_name
        self.description = description