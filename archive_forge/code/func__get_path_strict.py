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
def _get_path_strict(self, path: str) -> PathItem:
    path_item = self._paths_strict.get(path)
    if not path_item:
        raise ValueError(f'No path found for {path}')
    return path_item