from __future__ import annotations
import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import requests
import yaml
from langchain_core.pydantic_v1 import ValidationError
@property
def _parameters_strict(self) -> Dict[str, Union[Parameter, Reference]]:
    """Get parameters or err."""
    parameters = self._components_strict.parameters
    if parameters is None:
        raise ValueError('No parameters found in spec. ')
    return parameters