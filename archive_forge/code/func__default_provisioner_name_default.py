import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
@default('default_provisioner_name')
def _default_provisioner_name_default(self) -> str:
    """The default provisioner name."""
    return getenv(self.default_provisioner_name_env, 'local-provisioner')