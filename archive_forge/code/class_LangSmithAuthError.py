import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
class LangSmithAuthError(LangSmithError):
    """Couldn't authenticate with the LangSmith API."""