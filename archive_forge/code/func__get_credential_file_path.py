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
def _get_credential_file_path() -> str:
    return file_util.get_streamlit_file_path('credentials.toml')