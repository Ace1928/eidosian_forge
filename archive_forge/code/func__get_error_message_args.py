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
def _get_error_message_args(orig_exc: BaseException, failed_obj: Any) -> dict[str, Any]:
    hash_reason = hash_stacks.current.hash_reason
    hash_source = hash_stacks.current.hash_source
    failed_obj_type_str = type_util.get_fqn_type(failed_obj)
    object_part = ''
    if hash_source is None or hash_reason is None:
        object_desc = 'something'
    elif hash_reason is HashReason.CACHING_BLOCK:
        object_desc = 'a code block'
    else:
        if hasattr(hash_source, '__name__'):
            object_desc = f'`{hash_source.__name__}()`'
        else:
            object_desc = 'a function'
        if hash_reason is HashReason.CACHING_FUNC_ARGS:
            object_part = 'the arguments of'
        elif hash_reason is HashReason.CACHING_FUNC_BODY:
            object_part = 'the body of'
        elif hash_reason is HashReason.CACHING_FUNC_OUTPUT:
            object_part = 'the return value of'
    return {'orig_exception_desc': str(orig_exc), 'failed_obj_type_str': failed_obj_type_str, 'hash_stack': hash_stacks.current.pretty_print(), 'object_desc': object_desc, 'object_part': object_part}