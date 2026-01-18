import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from itertools import chain, filterfalse
from pathlib import Path
from threading import Timer
from typing import List, NamedTuple, Optional
import importlib_metadata
import pkg_resources  # noqa: TID251
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
import mlflow
from mlflow.environment_variables import MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils.versioning import _strip_dev_version_suffix
from mlflow.utils.databricks_utils import (
def _parse_requirements(requirements, is_constraint, base_dir=None):
    """A simplified version of `pip._internal.req.parse_requirements` which performs the following
    operations on the given requirements file and yields the parsed requirements.

    - Remove comments and blank lines
    - Join continued lines
    - Resolve requirements file references (e.g. '-r requirements.txt')
    - Resolve constraints file references (e.g. '-c constraints.txt')

    Args:
        requirements: A string path to a requirements file on the local filesystem or
            an iterable of pip requirement strings.
        is_constraint: Indicates the parsed requirements file is a constraint file.
        base_dir: If specified, resolve relative file references (e.g. '-r requirements.txt')
            against the specified directory.

    Returns:
        A list of ``_Requirement`` instances.

    References:
    - `pip._internal.req.parse_requirements`:
      https://github.com/pypa/pip/blob/7a77484a492c8f1e1f5ef24eaf71a43df9ea47eb/src/pip/_internal/req/req_file.py#L118
    - Requirements File Format:
      https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format
    - Constraints Files:
      https://pip.pypa.io/en/stable/user_guide/#constraints-files
    """
    if base_dir is None:
        if isinstance(requirements, (str, Path)):
            base_dir = os.path.dirname(requirements)
            with open(requirements) as f:
                requirements = f.read().splitlines()
        else:
            base_dir = os.getcwd()
    lines = map(str.strip, requirements)
    lines = map(_strip_inline_comment, lines)
    lines = _join_continued_lines(lines)
    lines = filterfalse(_is_comment, lines)
    lines = filterfalse(_is_empty, lines)
    for line in lines:
        if _is_requirements_file(line):
            req_file = line.split(maxsplit=1)[1]
            abs_path = os.path.join(base_dir, req_file)
            yield from _parse_requirements(abs_path, is_constraint=False)
        elif _is_constraints_file(line):
            req_file = line.split(maxsplit=1)[1]
            abs_path = os.path.join(base_dir, req_file)
            yield from _parse_requirements(abs_path, is_constraint=True)
        else:
            yield _Requirement(line, is_constraint)