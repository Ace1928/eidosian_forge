import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
def _validate_extension(data):
    """Detect if a package is an extension using its metadata.

    Returns any problems it finds.
    """
    jlab = data.get('jupyterlab', None)
    if jlab is None:
        return ['No `jupyterlab` key']
    if not isinstance(jlab, dict):
        return ['The `jupyterlab` key must be a JSON object']
    extension = jlab.get('extension', False)
    mime_extension = jlab.get('mimeExtension', False)
    theme_path = jlab.get('themePath', '')
    schema_dir = jlab.get('schemaDir', '')
    messages = []
    if not extension and (not mime_extension):
        messages.append('No `extension` or `mimeExtension` key present')
    if extension == mime_extension:
        msg = '`mimeExtension` and `extension` must point to different modules'
        messages.append(msg)
    files = data['jupyterlab_extracted_files']
    main = data.get('main', 'index.js')
    if not main.endswith('.js'):
        main += '.js'
    if extension is True:
        extension = main
    elif extension and (not extension.endswith('.js')):
        extension += '.js'
    if mime_extension is True:
        mime_extension = main
    elif mime_extension and (not mime_extension.endswith('.js')):
        mime_extension += '.js'
    if extension and extension not in files:
        messages.append('Missing extension module "%s"' % extension)
    if mime_extension and mime_extension not in files:
        messages.append('Missing mimeExtension module "%s"' % mime_extension)
    if theme_path and (not any((f.startswith(str(Path(theme_path))) for f in files))):
        messages.append('themePath is empty: "%s"' % theme_path)
    if schema_dir and (not any((f.startswith(str(Path(schema_dir))) for f in files))):
        messages.append('schemaDir is empty: "%s"' % schema_dir)
    return messages