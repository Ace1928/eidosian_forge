from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import operation_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
def IsGlobalUrlMapRef(url_map_ref):
    """Returns True if the URL Map reference is global."""
    return url_map_ref.Collection() == 'compute.urlMaps'