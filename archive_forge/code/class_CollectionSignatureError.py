from __future__ import (absolute_import, division, print_function)
import errno
import fnmatch
import functools
import json
import os
import pathlib
import queue
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import typing as t
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, fields as dc_fields
from hashlib import sha256
from io import BytesIO
from importlib.metadata import distribution
from itertools import chain
import ansible.constants as C
from ansible.compat.importlib_resources import files
from ansible.errors import AnsibleError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.galaxy_api_proxy import MultiGalaxyAPIProxy
from ansible.galaxy.collection.gpg import (
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import meets_requirements
from ansible.plugins.loader import get_all_plugin_loaders
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_dump
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash, secure_hash_s
from ansible.utils.sentinel import Sentinel
class CollectionSignatureError(Exception):

    def __init__(self, reasons=None, stdout=None, rc=None, ignore=False):
        self.reasons = reasons
        self.stdout = stdout
        self.rc = rc
        self.ignore = ignore
        self._reason_wrapper = None

    def _report_unexpected(self, collection_name):
        return f"Unexpected error for '{collection_name}': GnuPG signature verification failed with the return code {self.rc} and output {self.stdout}"

    def _report_expected(self, collection_name):
        header = f"Signature verification failed for '{collection_name}' (return code {self.rc}):"
        return header + self._format_reasons()

    def _format_reasons(self):
        if self._reason_wrapper is None:
            self._reason_wrapper = textwrap.TextWrapper(initial_indent='    * ', subsequent_indent='      ')
        wrapped_reasons = ['\n'.join(self._reason_wrapper.wrap(reason)) for reason in self.reasons]
        return '\n' + '\n'.join(wrapped_reasons)

    def report(self, collection_name):
        if self.reasons:
            return self._report_expected(collection_name)
        return self._report_unexpected(collection_name)