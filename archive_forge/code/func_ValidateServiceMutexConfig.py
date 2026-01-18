from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateServiceMutexConfig(unused_ref, unused_args, req):
    """Validates that the mutual exclusive configurations of Dataproc Metastore service are not set at the same time.

  Args:
    req: A request with `service` field.

  Returns:
    A request without service mutex configuration conflicts.
  Raises:
    BadArgumentException: when mutual exclusive configurations of service are
    set at the same time.
  """
    if req.service.encryptionConfig and req.service.encryptionConfig.kmsKey and req.service.metadataIntegration.dataCatalogConfig.enabled:
        raise exceptions.BadArgumentException('--data-catalog-sync', 'Data Catalog synchronization cannot be used in conjunction with customer-managed encryption keys.')
    return ValidateServiceMutexConfigForV1(unused_ref, unused_args, req)