from __future__ import annotations
import types
from typing import Any
from streamlit import type_util
from streamlit.errors import (
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
class UnhashableParamError(StreamlitAPIException):

    def __init__(self, cache_type: CacheType, func: types.FunctionType, arg_name: str | None, arg_value: Any, orig_exc: BaseException):
        msg = self._create_message(cache_type, func, arg_name, arg_value)
        super().__init__(msg)
        self.with_traceback(orig_exc.__traceback__)

    @staticmethod
    def _create_message(cache_type: CacheType, func: types.FunctionType, arg_name: str | None, arg_value: Any) -> str:
        arg_name_str = arg_name if arg_name is not None else '(unnamed)'
        arg_type = type_util.get_fqn_type(arg_value)
        func_name = func.__name__
        arg_replacement_name = f'_{arg_name}' if arg_name is not None else '_arg'
        return f"\nCannot hash argument '{arg_name_str}' (of type `{arg_type}`) in '{func_name}'.\n\nTo address this, you can tell Streamlit not to hash this argument by adding a\nleading underscore to the argument's name in the function signature:\n\n```\n@st.{get_decorator_api_name(cache_type)}\ndef {func_name}({arg_replacement_name}, ...):\n    ...\n```\n            ".strip('\n')