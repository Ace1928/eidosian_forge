from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def UpdateServiceMaskHook(unused_ref, args, update_service_req):
    """Constructs updateMask for update requests of Dataproc Metastore services.

  Args:
    unused_ref: A resource ref to the parsed Service resource.
    args: The parsed args namespace from CLI.
    update_service_req: Created Update request for the API call.

  Returns:
    Modified request for the API call.
  """
    update_service_req.updateMask = _GenerateUpdateMask(args)
    return update_service_req