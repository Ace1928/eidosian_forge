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
def _directory_archive(self, export_name, files, root):
    output_dir = os.path.join(root, self._unique_name(export_name, '', root)[0])
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for (basename, ext), entry in files:
        filename = self._truncate_name(basename, ext)
        fpath = os.path.join(output_dir, filename)
        with open(fpath, 'wb') as f:
            f.write(Exporter.encode(entry))