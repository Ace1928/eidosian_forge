from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
class GcsNotFoundError(GcsApiError, NotFoundError):
    """Error raised when the requested GCS resource does not exist.

  Implements custom formatting to avoid messy default.
  """

    def __init__(self, error, *args, **kwargs):
        del args, kwargs
        super(GcsNotFoundError, self).__init__(error, error_format='HTTPError {status_code}: {status_message}')
        if not error.url:
            return
        custom_error_format_for_buckets_and_objects = 'gs://{instance_name} not found: {status_code}.'
        _, _, resource_path = resource.SplitEndpointUrl(error.url)
        match_object_resource_path = OBJECT_RESOURCE_PATH_PATTERN.search(resource_path)
        if match_object_resource_path:
            self._custom_format_object_error(match_object_resource_path, custom_error_format_for_buckets_and_objects)
            return
        match_bucket_resource_path = BUCKET_RESOURCE_PATH_PATTERN.search(resource_path)
        if match_bucket_resource_path:
            self._custom_format_bucket_error(match_bucket_resource_path, custom_error_format_for_buckets_and_objects)

    def _custom_format_bucket_error(self, match_bucket_resource_path, error_format):
        """Sets custom error formatting for buckets resource paths.

    Args:
      match_bucket_resource_path (re.Match): Match object that contains the
        result of searching regex BUCKET_RESOURCE_PATH_PATTERN in a resource
        path.
      error_format (str): Custom error format for buckets.
    """
        self.error_format = error_format
        self.payload.instance_name = match_bucket_resource_path.group('bucket')

    def _custom_format_object_error(self, match_object_resource_path, error_format):
        """Sets custom error formatting for object resource paths.

    Args:
      match_object_resource_path (re.Match): Match object
        that contains the result of searching regex OBJECT_RESOURCE_PATH_PATTERN
        in a resource path.
      error_format (str): Custom error format for objects.
    """
        resource_path = match_object_resource_path.string
        params = urllib.parse.parse_qs(resource_path)
        if 'generation' in params:
            generation_string = '#' + params['generation'][0]
        else:
            generation_string = ''
        self.error_format = error_format
        self.payload.instance_name = '{}/{}{}'.format(match_object_resource_path.group('bucket'), match_object_resource_path.group('object'), generation_string)