import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union
from traitlets.config import Instance, LoggingConfigurable, Unicode
from ..connect import KernelConnectionInfo
def __apply_env_substitutions(self, substitution_values: Dict[str, str]) -> Dict[str, str]:
    """
        Walks entries in the kernelspec's env stanza and applies substitutions from current env.

        This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
        start sequence.

        Returns the substituted list of env entries.

        NOTE: This method is private and is not intended to be overridden by provisioners.
        """
    substituted_env = {}
    if self.kernel_spec:
        from string import Template
        templated_env = self.kernel_spec.env
        for k, v in templated_env.items():
            substituted_env.update({k: Template(v).safe_substitute(substitution_values)})
    return substituted_env