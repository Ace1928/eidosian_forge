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
def _get_root_referenced_schema(self, ref: Reference) -> Schema:
    """Get the root reference or err."""
    from openapi_pydantic import Reference
    schema = self.get_referenced_schema(ref)
    while isinstance(schema, Reference):
        schema = self.get_referenced_schema(schema)
    return schema