from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID, uuid4
import urllib.parse
from langsmith import schemas as ls_schemas
from langsmith import utils
from langsmith.client import ID_TYPE, RUN_TYPE_T, Client, _dumps_json
def _get_dicts_safe(self):
    try:
        return self.dict(exclude={'child_runs'}, exclude_none=True)
    except TypeError:
        self_dict = self.dict(exclude={'child_runs', 'inputs', 'outputs'}, exclude_none=True)
        if self.inputs:
            self_dict['inputs'] = self.inputs.copy()
        if self.outputs:
            self_dict['outputs'] = self.outputs.copy()
        return self_dict