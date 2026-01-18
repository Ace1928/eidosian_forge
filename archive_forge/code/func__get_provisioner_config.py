import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
def _get_provisioner_config(self, kernel_spec: Any) -> Dict[str, Any]:
    """
        Return the kernel_provisioner stanza from the kernel_spec.

        Checks the kernel_spec's metadata dictionary for a kernel_provisioner entry.
        If found, it is returned, else one is created relative to the DEFAULT_PROVISIONER
        and returned.

        Parameters
        ----------
        kernel_spec : Any - this is a KernelSpec type but listed as Any to avoid circular import
            The kernel specification object from which the provisioner dictionary is derived.

        Returns
        -------
        dict
            The provisioner portion of the kernel_spec.  If one does not exist, it will contain
            the default information.  If no `config` sub-dictionary exists, an empty `config`
            dictionary will be added.
        """
    env_provisioner = kernel_spec.metadata.get('kernel_provisioner', {})
    if 'provisioner_name' in env_provisioner:
        if 'config' not in env_provisioner:
            env_provisioner.update({'config': {}})
        return env_provisioner
    return {'provisioner_name': self.default_provisioner_name, 'config': {}}