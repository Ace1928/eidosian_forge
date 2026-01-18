import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@DeveloperAPI
class Tee(object):

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self._handling_warning = False

    def _warn(self, op, s, args, kwargs):
        if self._handling_warning:
            return
        msg = f"ValueError when calling '{op}' on stream ({s}). "
        msg += f'args: {args} kwargs: {kwargs}'
        self._handling_warning = True
        logger.warning(msg)
        self._handling_warning = False

    def seek(self, *args, **kwargs):
        for s in [self.stream1, self.stream2]:
            try:
                s.seek(*args, **kwargs)
            except ValueError:
                self._warn('seek', s, args, kwargs)

    def write(self, *args, **kwargs):
        for s in [self.stream1, self.stream2]:
            try:
                s.write(*args, **kwargs)
            except ValueError:
                self._warn('write', s, args, kwargs)

    def flush(self, *args, **kwargs):
        for s in [self.stream1, self.stream2]:
            try:
                s.flush(*args, **kwargs)
            except ValueError:
                self._warn('flush', s, args, kwargs)

    @property
    def encoding(self):
        if hasattr(self.stream1, 'encoding'):
            return self.stream1.encoding
        return self.stream2.encoding

    @property
    def error(self):
        if hasattr(self.stream1, 'error'):
            return self.stream1.error
        return self.stream2.error

    @property
    def newlines(self):
        if hasattr(self.stream1, 'newlines'):
            return self.stream1.newlines
        return self.stream2.newlines

    def detach(self):
        raise NotImplementedError

    def read(self, *args, **kwargs):
        raise NotImplementedError

    def readline(self, *args, **kwargs):
        raise NotImplementedError

    def tell(self, *args, **kwargs):
        raise NotImplementedError