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
def _add_content(self, obj, data, info, filename=None):
    unique_key, ext = self._compute_filename(obj, info, filename=filename)
    self._files[unique_key, ext] = (data, info)