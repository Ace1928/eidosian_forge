import dataclasses
import json
import logging
import os
import platform
from typing import Any, Dict, Optional
import torch
class xFormersInvalidLibException(Exception):

    def __init__(self, build_info: Optional[_BuildInfo]) -> None:
        self.build_info = build_info

    def __str__(self) -> str:
        if self.build_info is None:
            msg = 'xFormers was built for a different version of PyTorch or Python.'
        else:
            msg = f'xFormers was built for:\n    PyTorch {self.build_info.torch_version} with CUDA {self.build_info.cuda_version} (you have {torch.__version__})\n    Python  {self.build_info.python_version} (you have {platform.python_version()})'
        return "xFormers can't load C++/CUDA extensions. " + msg + '\n  Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)\n' + UNAVAILABLE_FEATURES_MSG