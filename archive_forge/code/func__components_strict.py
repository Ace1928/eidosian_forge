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
def _components_strict(self) -> Components:
    """Get components or err."""
    if self.components is None:
        raise ValueError('No components found in spec. ')
    return self.components