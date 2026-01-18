import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
def _process_templates(self, tmp):
    self.templates = {}
    if self.template_overrides is not None:
        tmp.update(self.template_overrides)
    for k, v in tmp.items():
        if '{{' in v:
            import jinja2
            self.templates[k] = lambda temp=v, **kwargs: jinja2.Template(temp).render(**kwargs)
        else:
            self.templates[k] = v