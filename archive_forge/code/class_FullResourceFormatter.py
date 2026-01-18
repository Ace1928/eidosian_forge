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
class FullResourceFormatter(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for a formatter to format the Resource object.

  This FullResourceFormatter is specifically used for ls -L output formatting.
  """

    def format_bucket(self, bucket_resource):
        """Returns a formatted string representing the BucketResource.

    Args:
      bucket_resource (resource_reference.BucketResource): A BucketResource
        instance.

    Returns:
      Formatted multi-line string representing the BucketResource.
    """
        raise NotImplementedError('format_bucket must be overridden.')

    def format_object(self, object_resource, show_acl=True, show_version_in_url=False, **kwargs):
        """Returns a formatted string representing the ObjectResource.

    Args:
      object_resource (resource_reference.Resource): A Resource instance.
      show_acl (bool): Include ACLs list in resource display.
      show_version_in_url (bool): Display extended URL with versioning info.
      **kwargs (dict): Unused. May apply to other resource format functions.

    Returns:
      Formatted multi-line string represnting the ObjectResource.
    """
        raise NotImplementedError('format_object must be overridden.')

    def format(self, resource, **kwargs):
        """Type-checks resource and returns a formatted metadata string."""
        if isinstance(resource, resource_reference.BucketResource):
            return self.format_bucket(resource)
        if isinstance(resource, resource_reference.ObjectResource):
            return self.format_object(resource, **kwargs)
        raise NotImplementedError('{} does not support {}'.format(self.__class__, type(resource)))