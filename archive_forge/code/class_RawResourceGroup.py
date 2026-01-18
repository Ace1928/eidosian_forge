from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
class RawResourceGroup:
    """Representation of the raw ResourceGroup output from kubectl."""

    def __init__(self, cluster, rg_dict):
        """Initialize a RawResourceGroup object.

    Args:
      cluster: name of the cluster the results are from
      rg_dict: raw ResourceGroup dictionary parsed from kubectl
    """
        self.cluster = cluster
        self.rg_dict = rg_dict