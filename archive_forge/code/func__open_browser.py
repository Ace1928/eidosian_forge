import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, cast
from langsmith import env as ls_env
from langsmith import utils as ls_utils
def _open_browser(self, url: str) -> None:
    try:
        subprocess.run(['open', url])
    except FileNotFoundError:
        pass