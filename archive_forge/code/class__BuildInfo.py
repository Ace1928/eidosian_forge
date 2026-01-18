import dataclasses
import json
import logging
import os
import platform
from typing import Any, Dict, Optional
import torch
@dataclasses.dataclass
class _BuildInfo:
    metadata: Dict[str, Any]

    @property
    def cuda_version(self) -> Optional[int]:
        return self.metadata['version']['cuda']

    @property
    def hip_version(self) -> Optional[int]:
        return self.metadata['version']['hip']

    @property
    def torch_version(self) -> str:
        return self.metadata['version']['torch']

    @property
    def python_version(self) -> str:
        return self.metadata['version']['python']

    @property
    def flash_version(self) -> str:
        return self.metadata['version'].get('flash', '0.0.0')

    @property
    def build_env(self) -> Dict[str, Any]:
        return self.metadata['env']