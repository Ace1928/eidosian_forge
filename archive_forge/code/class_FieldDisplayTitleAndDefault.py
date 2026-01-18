from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
import six
class FieldDisplayTitleAndDefault(object):
    """Holds the title and default value to be displayed for a resource field."""

    def __init__(self, title, default, field_name=None):
        """Initializes FieldDisplayTitleAndDefault.

    Args:
      title (str): The title for the field.
      default (str): The default value to be used if value is missing.
      field_name (str|None): The field name to be used to extract
        the data from Resource object.
        If None, the field name from BucketDisplayTitlesAndDefaults or
        ObjectDisplayTitlesAndDefaults is used.
    """
        self.title = title
        self.default = default
        self.field_name = field_name