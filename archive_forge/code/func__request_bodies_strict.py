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
def _request_bodies_strict(self) -> Dict[str, Union[RequestBody, Reference]]:
    """Get the request body or err."""
    request_bodies = self._components_strict.requestBodies
    if request_bodies is None:
        raise ValueError('No request body found in spec. ')
    return request_bodies