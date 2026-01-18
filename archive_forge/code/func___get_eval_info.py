import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def __get_eval_info(self) -> None:
    """Get inner evaluation count and names."""
    if self.__need_reload_eval_info:
        self.__need_reload_eval_info = False
        out_num_eval = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetEvalCounts(self._handle, ctypes.byref(out_num_eval)))
        self.__num_inner_eval = out_num_eval.value
        if self.__num_inner_eval > 0:
            tmp_out_len = ctypes.c_int(0)
            reserved_string_buffer_size = 255
            required_string_buffer_size = ctypes.c_size_t(0)
            string_buffers = [ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(self.__num_inner_eval)]
            ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
            _safe_call(_LIB.LGBM_BoosterGetEvalNames(self._handle, ctypes.c_int(self.__num_inner_eval), ctypes.byref(tmp_out_len), ctypes.c_size_t(reserved_string_buffer_size), ctypes.byref(required_string_buffer_size), ptr_string_buffers))
            if self.__num_inner_eval != tmp_out_len.value:
                raise ValueError("Length of eval names doesn't equal with num_evals")
            actual_string_buffer_size = required_string_buffer_size.value
            if reserved_string_buffer_size < actual_string_buffer_size:
                string_buffers = [ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(self.__num_inner_eval)]
                ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
                _safe_call(_LIB.LGBM_BoosterGetEvalNames(self._handle, ctypes.c_int(self.__num_inner_eval), ctypes.byref(tmp_out_len), ctypes.c_size_t(actual_string_buffer_size), ctypes.byref(required_string_buffer_size), ptr_string_buffers))
            self.__name_inner_eval = [string_buffers[i].value.decode('utf-8') for i in range(self.__num_inner_eval)]
            self.__higher_better_inner_eval = [name.startswith(('auc', 'ndcg@', 'map@', 'average_precision')) for name in self.__name_inner_eval]