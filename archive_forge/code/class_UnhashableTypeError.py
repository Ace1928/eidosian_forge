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
class UnhashableTypeError(StreamlitAPIException):

    def __init__(self, orig_exc, failed_obj):
        msg = self._get_message(orig_exc, failed_obj)
        super().__init__(msg)
        self.with_traceback(orig_exc.__traceback__)

    def _get_message(self, orig_exc, failed_obj):
        args = _get_error_message_args(orig_exc, failed_obj)
        return ("\nCannot hash object of type `%(failed_obj_type_str)s`, found in %(object_part)s\n%(object_desc)s.\n\nWhile caching %(object_part)s %(object_desc)s, Streamlit encountered an\nobject of type `%(failed_obj_type_str)s`, which it does not know how to hash.\n\nTo address this, please try helping Streamlit understand how to hash that type\nby passing the `hash_funcs` argument into `@st.cache`. For example:\n\n```\n@st.cache(hash_funcs={%(failed_obj_type_str)s: my_hash_func})\ndef my_func(...):\n    ...\n```\n\nIf you don't know where the object of type `%(failed_obj_type_str)s` is coming\nfrom, try looking at the hash chain below for an object that you do recognize,\nthen pass that to `hash_funcs` instead:\n\n```\n%(hash_stack)s\n```\n\nPlease see the `hash_funcs` [documentation](https://docs.streamlit.io/library/advanced-features/caching#the-hash_funcs-parameter)\nfor more details.\n            " % args).strip('\n')