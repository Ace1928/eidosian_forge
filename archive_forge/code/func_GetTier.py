from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def GetTier():
    """Returns the value of the metastore/tier config property.

  Config properties can be overridden with command line flags. If the --tier
  flag was provided, this will return the value provided with the flag.
  """
    return properties.VALUES.metastore.tier.Get(required=True)