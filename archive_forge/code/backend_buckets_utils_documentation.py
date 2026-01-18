from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
Applies the CdnPolicy arguments to the specified backend bucket.

  If there are no arguments related to CdnPolicy, the backend bucket remains
  unmodified.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_bucket: The backend bucket object.
    is_update: True if this is called on behalf of an update command instead of
      a create command, False otherwise.
    cleared_fields: Reference to list with fields that should be cleared. Valid
      only for update command.
  