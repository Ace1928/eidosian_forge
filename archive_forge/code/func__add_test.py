from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
def _add_test(self, category: str, test_name: str, test_func: Callable[..., Any], report_item: list[ModelProto | NodeProto | None], devices: Iterable[str]=('CPU', 'CUDA'), **kwargs: Any) -> None:
    if not test_name.startswith('test_'):
        raise ValueError(f'Test name must start with test_: {test_name}')

    def add_device_test(device: str) -> None:
        device_test_name = f'{test_name}_{device.lower()}'
        if device_test_name in self._test_items[category]:
            raise ValueError(f'Duplicated test name "{device_test_name}" in category "{category}"')

        @unittest.skipIf(not self.backend.supports_device(device), f"Backend doesn't support device {device}")
        @functools.wraps(test_func)
        def device_test_func(*args: Any, **device_test_kwarg: Any) -> Any:
            try:
                merged_kwargs = {**kwargs, **device_test_kwarg}
                return test_func(*args, device, **merged_kwargs)
            except BackendIsNotSupposedToImplementIt as e:
                if '-v' in sys.argv or '--verbose' in sys.argv:
                    print(f'Test {device_test_name} is effectively skipped: {e}')
        self._test_items[category][device_test_name] = TestItem(device_test_func, report_item)
    for device in devices:
        add_device_test(device)