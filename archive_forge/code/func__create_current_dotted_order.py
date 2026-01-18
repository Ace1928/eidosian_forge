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
def _create_current_dotted_order(start_time: Optional[datetime], run_id: Optional[UUID]) -> str:
    """Create the current dotted order."""
    st = start_time or datetime.now(timezone.utc)
    id_ = run_id or uuid4()
    return st.strftime('%Y%m%dT%H%M%S%fZ') + str(id_)