from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.resource import yaml_printer
Printer for k8s_objects to export.

  Omits status information, and metadata that isn't consistent across
  deployments, like project or region.
  