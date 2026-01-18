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
def __create_connection(name: str, connection_class: type[ConnectionClass], **kwargs) -> ConnectionClass:
    return connection_class(connection_name=name, **kwargs)