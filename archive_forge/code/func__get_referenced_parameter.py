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
def _get_referenced_parameter(self, ref: Reference) -> Union[Parameter, Reference]:
    """Get a parameter (or nested reference) or err."""
    ref_name = ref.ref.split('/')[-1]
    parameters = self._parameters_strict
    if ref_name not in parameters:
        raise ValueError(f'No parameter found for {ref_name}')
    return parameters[ref_name]