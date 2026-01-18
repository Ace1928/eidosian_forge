from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import os
from typing import Any
from googlecloudsdk.api_lib.container import kubeconfig as container_kubeconfig
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
def Cluster(name, server):
    """Generate and return a cluster kubeconfig object."""
    cluster = {'server': server}
    return {'name': name, 'cluster': cluster}