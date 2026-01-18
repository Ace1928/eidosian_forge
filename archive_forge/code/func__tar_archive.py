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
def _tar_archive(self, export_name, files, root):
    archname = '.'.join(self._unique_name(export_name, 'tar', root))
    with tarfile.TarFile(os.path.join(root, archname), 'w') as tarf:
        for (basename, ext), entry in files:
            filename = self._truncate_name(basename, ext)
            tarinfo = tarfile.TarInfo(f'{export_name}/{filename}')
            filedata = Exporter.encode(entry)
            tarinfo.size = len(filedata)
            tarf.addfile(tarinfo, BytesIO(filedata))