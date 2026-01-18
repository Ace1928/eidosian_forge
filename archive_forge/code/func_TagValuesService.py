from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def TagValuesService():
    """Returns the tag values service class."""
    client = TagClient()
    return client.tagValues