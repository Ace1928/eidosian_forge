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
def _get_root_referenced_parameter(self, ref: Reference) -> Parameter:
    """Get the root reference or err."""
    from openapi_pydantic import Reference
    parameter = self._get_referenced_parameter(ref)
    while isinstance(parameter, Reference):
        parameter = self._get_referenced_parameter(parameter)
    return parameter