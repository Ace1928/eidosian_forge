import itertools
import os
import pickle
import re
import shutil
import string
import tarfile
import time
import zipfile
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import param
from param.parameterized import bothmethod
from .dimension import LabelledData
from .element import Collator, Element
from .ndmapping import NdMapping, UniformNdMapping
from .options import Store
from .overlay import Layout, Overlay
from .util import group_sanitizer, label_sanitizer, unique_iterator
def _dim_formatter(self, obj):
    if not obj:
        return ''
    key_dims = obj.traverse(lambda x: x.kdims, [UniformNdMapping])
    constant_dims = obj.traverse(lambda x: x.cdims)
    dims = []
    map(dims.extend, key_dims + constant_dims)
    dims = unique_iterator(dims)
    dim_strings = []
    for dim in dims:
        lower, upper = obj.range(dim.name)
        lower, upper = (dim.pprint_value(lower), dim.pprint_value(upper))
        if lower == upper:
            range = dim.pprint_value(lower)
        else:
            range = f'{lower}-{upper}'
        formatters = {'name': dim.name, 'range': range, 'unit': dim.unit}
        dim_strings.append(self.dimension_formatter.format(**formatters))
    return '_'.join(dim_strings)