from __future__ import annotations
import os
import re
from datetime import timedelta
from typing import Any, Final, Literal, TypeVar, overload
from streamlit.connections import (
from streamlit.deprecation_util import deprecate_obj_name
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_resource
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.secrets import secrets_singleton
def _get_first_party_connection(connection_class: str):
    if connection_class in FIRST_PARTY_CONNECTIONS:
        return FIRST_PARTY_CONNECTIONS[connection_class]
    raise StreamlitAPIException(f"Invalid connection '{connection_class}'. Supported connection classes: {FIRST_PARTY_CONNECTIONS}")