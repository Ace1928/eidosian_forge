from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
def GetPresetProfileOptions():
    """Returns the possible string options for the use-preset-profile flag."""
    return sorted(_PRESET_PROFILES.keys())