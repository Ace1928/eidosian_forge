import asyncio
import os
import signal
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from ..connect import KernelConnectionInfo, LocalPortCache
from ..launcher import launch_kernel
from ..localinterfaces import is_local_ip, local_ips
from .provisioner_base import KernelProvisionerBase
@staticmethod
def _scrub_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove any keyword arguments that Popen does not tolerate."""
    keywords_to_scrub: List[str] = ['extra_arguments', 'kernel_id']
    scrubbed_kwargs = kwargs.copy()
    for kw in keywords_to_scrub:
        scrubbed_kwargs.pop(kw, None)
    return scrubbed_kwargs