import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import srsly
from catalogue import RegistryError
from thinc.api import Config
from wasabi import MarkdownRenderer, Printer, get_raw_input
from .. import about, util
from ..compat import importlib_metadata
from ..schemas import ModelMetaSchema, validate
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app, string_to_list
import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
def _format_label_scheme(data: Dict[str, Any]) -> str:
    if not data:
        return ''
    md = MarkdownRenderer()
    n_labels = 0
    n_pipes = 0
    label_data = []
    for pipe, labels in data.items():
        if not labels:
            continue
        col1 = md.bold(md.code(pipe))
        col2 = ', '.join([md.code(str(label).replace('|', '\\|')) for label in labels])
        label_data.append((col1, col2))
        n_labels += len(labels)
        n_pipes += 1
    if not label_data:
        return ''
    label_info = f'View label scheme ({n_labels} labels for {n_pipes} components)'
    md.add('<details>')
    md.add(f'<summary>{label_info}</summary>')
    md.add(md.table(label_data, ['Component', 'Labels']))
    md.add('</details>')
    return md.text