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
def _get_message_from_func(self, orig_exc, cached_func, hash_func):
    args = _get_error_message_args(orig_exc, cached_func)
    if hasattr(hash_func, '__name__'):
        args['hash_func_name'] = '`%s()`' % hash_func.__name__
    else:
        args['hash_func_name'] = 'a function'
    return ("\n%(orig_exception_desc)s\n\nThis error is likely due to a bug in %(hash_func_name)s, which is a\nuser-defined hash function that was passed into the `@st.cache` decorator of\n%(object_desc)s.\n\n%(hash_func_name)s failed when hashing an object of type\n`%(failed_obj_type_str)s`.  If you don't know where that object is coming from,\ntry looking at the hash chain below for an object that you do recognize, then\npass that to `hash_funcs` instead:\n\n```\n%(hash_stack)s\n```\n\nIf you think this is actually a Streamlit bug, please\n[file a bug report here](https://github.com/streamlit/streamlit/issues/new/choose).\n            " % args).strip('\n')