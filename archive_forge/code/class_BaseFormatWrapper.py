from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
class BaseFormatWrapper(six.with_metaclass(abc.ABCMeta)):
    """For formatting how items are printed when listed.

  Child classes should set _header_wrapper and _object_wrapper.

  Attributes:
    resource (resource_reference.Resource): Item to be formatted for printing.
  """

    def __init__(self, resource, display_detail=DisplayDetail.SHORT, full_formatter=None, include_etag=None, object_state=None, readable_sizes=False, use_gsutil_style=False):
        """Initializes wrapper instance.

    Args:
      resource (resource_reference.Resource): Item to be formatted for printing.
      display_detail (DisplayDetail): Level of metadata detail for printing.
      full_formatter (base.FullResourceFormatter): Printing formatter used witch
        FULL DisplayDetail.
      include_etag (bool): Display etag string of resource.
      object_state (cloud_api.ObjectState): What versions of an object to query.
      readable_sizes (bool): Convert bytes to a more human readable format for
        long lising. For example, print 1024B as 1KiB.
      use_gsutil_style (bool): Outputs closer to the style of the gsutil CLI.
    """
        self.resource = resource
        self._display_detail = display_detail
        self._full_formatter = full_formatter
        self._include_etag = include_etag
        self._object_state = object_state
        self._readable_sizes = readable_sizes
        self._use_gsutil_style = use_gsutil_style

    def _check_and_handles_versions(self):
        show_version_in_url = self._object_state in (cloud_api.ObjectState.LIVE_AND_NONCURRENT, cloud_api.ObjectState.SOFT_DELETED)
        if show_version_in_url:
            url_string = self.resource.storage_url.url_string
            metageneration_string = '  metageneration={}'.format(six.text_type(self.resource.metageneration))
        else:
            url_string = self.resource.storage_url.versionless_url_string
            metageneration_string = ''
        return (url_string, metageneration_string)