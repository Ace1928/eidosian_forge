import os
import uuid
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def _GetObject(source, resource_ref):
    """Gets the object name for a source to be uploaded."""
    suffix = '.zip'
    if source.startswith('gs://') or os.path.isfile(source):
        _, suffix = os.path.splitext(source)
    file_name = '{stamp}-{uuid}{suffix}'.format(stamp=times.GetTimeStampFromDateTime(times.Now()), uuid=uuid.uuid4().hex, suffix=suffix)
    object_path = f'{_GetResourceType(resource_ref)}/{resource_ref.Name()}/{file_name}'
    return object_path