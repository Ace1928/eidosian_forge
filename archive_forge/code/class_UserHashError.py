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
class UserHashError(StreamlitAPIException):

    def __init__(self, orig_exc, cached_func_or_code, hash_func=None, lineno=None):
        self.alternate_name = type(orig_exc).__name__
        if hash_func:
            msg = self._get_message_from_func(orig_exc, cached_func_or_code, hash_func)
        else:
            msg = self._get_message_from_code(orig_exc, cached_func_or_code, lineno)
        super().__init__(msg)
        self.with_traceback(orig_exc.__traceback__)

    def _get_message_from_func(self, orig_exc, cached_func, hash_func):
        args = _get_error_message_args(orig_exc, cached_func)
        if hasattr(hash_func, '__name__'):
            args['hash_func_name'] = '`%s()`' % hash_func.__name__
        else:
            args['hash_func_name'] = 'a function'
        return ("\n%(orig_exception_desc)s\n\nThis error is likely due to a bug in %(hash_func_name)s, which is a\nuser-defined hash function that was passed into the `@st.cache` decorator of\n%(object_desc)s.\n\n%(hash_func_name)s failed when hashing an object of type\n`%(failed_obj_type_str)s`.  If you don't know where that object is coming from,\ntry looking at the hash chain below for an object that you do recognize, then\npass that to `hash_funcs` instead:\n\n```\n%(hash_stack)s\n```\n\nIf you think this is actually a Streamlit bug, please\n[file a bug report here](https://github.com/streamlit/streamlit/issues/new/choose).\n            " % args).strip('\n')

    def _get_message_from_code(self, orig_exc: BaseException, cached_code, lineno: int):
        args = _get_error_message_args(orig_exc, cached_code)
        failing_lines = _get_failing_lines(cached_code, lineno)
        failing_lines_str = ''.join(failing_lines)
        failing_lines_str = textwrap.dedent(failing_lines_str).strip('\n')
        args['failing_lines_str'] = failing_lines_str
        args['filename'] = cached_code.co_filename
        args['lineno'] = lineno
        return ('\n%(orig_exception_desc)s\n\nStreamlit encountered an error while caching %(object_part)s %(object_desc)s.\nThis is likely due to a bug in `%(filename)s` near line `%(lineno)s`:\n\n```\n%(failing_lines_str)s\n```\n\nPlease modify the code above to address this.\n\nIf you think this is actually a Streamlit bug, you may [file a bug report\nhere.] (https://github.com/streamlit/streamlit/issues/new/choose)\n        ' % args).strip('\n')