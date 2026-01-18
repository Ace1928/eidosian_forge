from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
import six
def GetRegistry(version):
    global registry
    try:
        resources.REGISTRY.GetCollectionInfo('appengine', version)
    except resources.InvalidCollectionException:
        registry = resources.REGISTRY.Clone()
        registry.RegisterApiByName('appengine', version)
    return registry