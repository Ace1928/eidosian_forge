from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
def CreatePreview(preview, preview_id, location):
    """Calls into the CreatePreview API.

  Args:
    preview: a messages.Preview resource (containing properties like the
      blueprint).
    preview_id: the ID of the preview, e.g. "my-preview" in
      "projects/p/locations/l/previews/my-preview".
    location: the location in which to create the preview.

  Returns:
    A messages.OperationMetadata representing a long-running operation.
  """
    client = GetClientInstance()
    messages = client.MESSAGES_MODULE
    return client.projects_locations_previews.Create(messages.ConfigProjectsLocationsPreviewsCreateRequest(parent=location, preview=preview, previewId=preview_id))