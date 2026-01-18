from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import path_simplifier
import six
@staticmethod
def _build_key(igm_ref, instance_ref):
    """Builds simple key object for combination of IGM and instance refs."""
    return (igm_ref, instance_ref)