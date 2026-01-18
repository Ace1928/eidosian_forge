import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
def _get_provisioner(self, name: str) -> EntryPoint:
    """Wrapper around entry_points (to fetch a single provisioner) - primarily to facilitate testing."""
    eps = entry_points(group=KernelProvisionerFactory.GROUP_NAME, name=name)
    if eps:
        return eps[0]
    if name == 'local-provisioner':
        distros = glob.glob(f'{path.dirname(path.dirname(__file__))}-*')
        self.log.warning(f"Kernel Provisioning: The 'local-provisioner' is not found.  This is likely due to the presence of multiple jupyter_client distributions and a previous distribution is being used as the source for entrypoints - which does not include 'local-provisioner'.  That distribution should be removed such that only the version-appropriate distribution remains (version >= 7).  Until then, a 'local-provisioner' entrypoint will be automatically constructed and used.\nThe candidate distribution locations are: {distros}")
        return EntryPoint('local-provisioner', 'jupyter_client.provisioning', 'LocalProvisioner')
    raise