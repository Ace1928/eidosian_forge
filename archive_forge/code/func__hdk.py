from typing import List, Optional, Tuple, Union
import pyarrow as pa
import pyhdk
from packaging import version
from pyhdk.hdk import HDK, ExecutionResult, QueryNode, RelAlgExecutor
from modin.config import CpuCount, HdkFragmentSize, HdkLaunchParameters
from modin.utils import _inherit_docstrings
from .base_worker import BaseDbWorker, DbTable
@classmethod
def _hdk(cls) -> HDK:
    """
        Initialize and return an HDK instance.

        Returns
        -------
        HDK
        """
    params = HdkLaunchParameters.get()
    cls._preferred_device = 'CPU' if params['cpu_only'] else 'GPU'
    cls._hdk_instance = HDK(**params)
    cls._hdk = cls._get_hdk_instance
    return cls._hdk()