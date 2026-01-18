from __future__ import annotations
import collections
import enum
import functools
import hashlib
import inspect
import io
import os
import pickle
import sys
import tempfile
import textwrap
import threading
import weakref
from typing import Any, Callable, Dict, Pattern, Type, Union
from streamlit import config, file_util, type_util, util
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.folder_black_list import FolderBlackList
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.util import HASHLIB_KWARGS
from, try looking at the hash chain below for an object that you do recognize,
from, try looking at the hash chain below for an object that you do recognize,
class InternalHashError(MarkdownFormattedException):
    """Exception in Streamlit hashing code (i.e. not a user error)"""

    def __init__(self, orig_exc: BaseException, failed_obj: Any):
        msg = self._get_message(orig_exc, failed_obj)
        super().__init__(msg)
        self.with_traceback(orig_exc.__traceback__)

    def _get_message(self, orig_exc: BaseException, failed_obj: Any) -> str:
        args = _get_error_message_args(orig_exc, failed_obj)
        return ("\n%(orig_exception_desc)s\n\nWhile caching %(object_part)s %(object_desc)s, Streamlit encountered an\nobject of type `%(failed_obj_type_str)s`, which it does not know how to hash.\n\n**In this specific case, it's very likely you found a Streamlit bug so please\n[file a bug report here.]\n(https://github.com/streamlit/streamlit/issues/new/choose)**\n\nIn the meantime, you can try bypassing this error by registering a custom\nhash function via the `hash_funcs` keyword in @st.cache(). For example:\n\n```\n@st.cache(hash_funcs={%(failed_obj_type_str)s: my_hash_func})\ndef my_func(...):\n    ...\n```\n\nIf you don't know where the object of type `%(failed_obj_type_str)s` is coming\nfrom, try looking at the hash chain below for an object that you do recognize,\nthen pass that to `hash_funcs` instead:\n\n```\n%(hash_stack)s\n```\n\nPlease see the `hash_funcs` [documentation](https://docs.streamlit.io/library/advanced-features/caching#the-hash_funcs-parameter)\nfor more details.\n            " % args).strip('\n')