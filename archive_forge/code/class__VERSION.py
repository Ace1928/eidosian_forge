from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class _VERSION(object):
    """An enum representing the API version of Deployment Manager."""

    def __init__(self, id, help_tag, help_note):
        self.id = id
        self.help_tag = help_tag
        self.help_note = help_note

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id