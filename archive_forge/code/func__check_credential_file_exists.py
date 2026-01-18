from __future__ import annotations
import json
import os
import sys
import textwrap
from collections import namedtuple
from datetime import datetime
from typing import Final, NoReturn
from uuid import uuid4
from streamlit import cli_util, env_util, file_util, util
from streamlit.logger import get_logger
def _check_credential_file_exists() -> bool:
    return os.path.exists(_get_credential_file_path())