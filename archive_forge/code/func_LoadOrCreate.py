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
@classmethod
def LoadOrCreate(cls, filename):
    """Read in the kubeconfig, and if it doesn't exist create one there."""
    try:
        return cls.LoadFromFile(filename)
    except (Error, IOError) as error:
        log.debug('unable to load default kubeconfig: {0}; recreating {1}'.format(error, filename))
        file_utils.MakeDir(os.path.dirname(filename))
        kubeconfig = cls(EmptyKubeconfig(), filename)
        kubeconfig.SaveToFile()
        return kubeconfig