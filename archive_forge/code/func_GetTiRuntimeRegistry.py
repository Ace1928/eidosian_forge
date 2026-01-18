from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import enum
from googlecloudsdk.api_lib.app import runtime_registry
def GetTiRuntimeRegistry():
    """A simple registry whose `Get()` method answers True if runtime is Ti."""
    return runtime_registry.Registry(_TI_RUNTIME_REGISTRY, default=False)