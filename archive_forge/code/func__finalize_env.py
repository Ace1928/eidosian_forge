import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union
from traitlets.config import Instance, LoggingConfigurable, Unicode
from ..connect import KernelConnectionInfo
def _finalize_env(self, env: Dict[str, str]) -> None:
    """
        Ensures env is appropriate prior to launch.

        This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
        start sequence.

        NOTE: Subclasses should be sure to call super()._finalize_env(env)
        """
    if self.kernel_spec.language and self.kernel_spec.language.lower().startswith('python'):
        env.pop('PYTHONEXECUTABLE', None)