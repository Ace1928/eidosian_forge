from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def EffectiveTagsService():
    """Returns the effective tags service class."""
    client = TagClient()
    return client.effectiveTags