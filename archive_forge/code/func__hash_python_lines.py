import inspect
import re
from typing import Dict, List, Tuple
from huggingface_hub.utils import insecure_hashlib
from .arrow import arrow
from .audiofolder import audiofolder
from .cache import cache  # noqa F401
from .csv import csv
from .imagefolder import imagefolder
from .json import json
from .pandas import pandas
from .parquet import parquet
from .sql import sql  # noqa F401
from .text import text
from .webdataset import webdataset
def _hash_python_lines(lines: List[str]) -> str:
    filtered_lines = []
    for line in lines:
        line = re.sub('#.*', '', line)
        if line:
            filtered_lines.append(line)
    full_str = '\n'.join(filtered_lines)
    full_bytes = full_str.encode('utf-8')
    return insecure_hashlib.sha256(full_bytes).hexdigest()