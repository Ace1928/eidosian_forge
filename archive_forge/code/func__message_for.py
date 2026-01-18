from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata
from io import StringIO
from json import JSONDecodeError
from pathlib import Path
from textwrap import dedent
from unittest import TestCase
import json
import os
import subprocess
import sys
import tempfile
import warnings
from jsonschema import Draft4Validator, Draft202012Validator
from jsonschema.exceptions import (
from jsonschema.validators import _LATEST_VERSION, validate
def _message_for(non_json):
    try:
        json.loads(non_json)
    except JSONDecodeError as error:
        return str(error)
    else:
        raise RuntimeError('Tried and failed to capture a JSON dump error.')