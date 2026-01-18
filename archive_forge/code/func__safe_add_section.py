import configparser
import getpass
import os
import tempfile
from typing import Any, Optional
from wandb import env
from wandb.old import core
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.runid import generate_id
@staticmethod
def _safe_add_section(settings, section):
    if not settings.has_section(section):
        settings.add_section(section)