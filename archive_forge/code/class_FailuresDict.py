import datetime
import difflib
import functools
import inspect
import json
import os
import re
import tempfile
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch._dynamo
import torch.utils._pytree as pytree
from torch._dynamo.utils import clone_input
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch._utils_internal import get_file_path_2
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
class FailuresDict:

    def __init__(self, path: str, data: FailuresDictData):
        self.path = path
        self.data = data

    @staticmethod
    def load(path, *, create_file=False) -> 'FailuresDict':
        if create_file and (not os.path.exists(path)):
            result = FailuresDict(path, {})
            FailuresDict.save()
            return result
        with open(path) as fp:
            contents = fp.read()
            if contents.strip() == '':
                dct = {'_description': DESCRIPTION, 'data': {}, '_version': VERSION}
            else:
                dct = json.loads(contents)
                assert 'data' in dct
                assert '_version' in dct and dct['_version'] == VERSION
        return FailuresDict(path, dct['data'])

    def _save(self, to_str=False) -> Optional[str]:
        to_dump = {'_description': DESCRIPTION, 'data': self.data, '_version': VERSION}
        serialized = json.dumps(to_dump, **DUMP_OPTIONS) + '\n'
        if to_str:
            return serialized
        with open(self.path, 'w') as fp:
            fp.write(serialized)
        return None

    def save(self) -> None:
        return self._save()

    def get_status(self, qualname: str, test_name: str) -> str:
        if qualname not in self.data:
            return 'xsuccess'
        dct = self.data[qualname]
        if test_name not in dct:
            return 'xsuccess'
        return dct[test_name]['status']

    def set_status(self, qualname: str, test_name: str, status: str, *, comment: Optional[str]=None):
        if qualname not in self.data:
            self.data[qualname] = {}
        dct = self.data[qualname]
        if test_name not in dct:
            dct[test_name] = {'status': None, 'comment': ''}
        if status == 'xsuccess':
            del dct[test_name]
        else:
            dct[test_name]['status'] = status
            if comment is not None:
                dct[test_name]['comment'] = comment