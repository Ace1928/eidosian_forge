from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
def _get_kernel_spec_by_name(self, kernel_name: str, resource_dir: str) -> KernelSpec:
    """Returns a :class:`KernelSpec` instance for a given kernel_name
        and resource_dir.
        """
    kspec = None
    if kernel_name == NATIVE_KERNEL_NAME:
        try:
            from ipykernel.kernelspec import RESOURCES, get_kernel_dict
        except ImportError:
            pass
        else:
            if resource_dir == RESOURCES:
                kdict = get_kernel_dict()
                kspec = self.kernel_spec_class(resource_dir=resource_dir, **kdict)
    if not kspec:
        kspec = self.kernel_spec_class.from_resource_dir(resource_dir)
    if not KPF.instance(parent=self.parent).is_provisioner_available(kspec):
        raise NoSuchKernel(kernel_name)
    return kspec