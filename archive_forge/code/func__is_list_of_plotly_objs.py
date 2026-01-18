from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def _is_list_of_plotly_objs(obj: object) -> TypeGuard[list[Any]]:
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return False
    return all((_is_plotly_obj(item) for item in obj))