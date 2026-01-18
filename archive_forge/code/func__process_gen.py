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
def _process_gen(self, gens):
    out = {}
    for gen in gens:
        dimension = {k: v if isinstance(v, list) else range(v.get('start', 0), v['stop'], v.get('step', 1)) for k, v in gen['dimensions'].items()}
        products = (dict(zip(dimension.keys(), values)) for values in itertools.product(*dimension.values()))
        for pr in products:
            import jinja2
            key = jinja2.Template(gen['key']).render(**pr, **self.templates)
            url = jinja2.Template(gen['url']).render(**pr, **self.templates)
            if 'offset' in gen and 'length' in gen:
                offset = int(jinja2.Template(gen['offset']).render(**pr, **self.templates))
                length = int(jinja2.Template(gen['length']).render(**pr, **self.templates))
                out[key] = [url, offset, length]
            elif ('offset' in gen) ^ ('length' in gen):
                raise ValueError("Both 'offset' and 'length' are required for a reference generator entry if either is provided.")
            else:
                out[key] = [url]
    return out