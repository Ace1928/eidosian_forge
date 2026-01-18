from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseCloudKmsKey(cloud_kms_key):
    """Parses a Cloud KMS key using configuration properties for fallback.

  Args:
    cloud_kms_key: str, fully-qualified URL, or relative name

  Returns:
    str: the relative name of the Cloud KMS key resource
  """
    return resources.REGISTRY.Parse(cloud_kms_key, collection='cloudkms.projects.locations.keyRings.cryptoKeys').RelativeName()