import configparser
import getpass
import os
import tempfile
from typing import Any, Optional
from wandb import env
from wandb.old import core
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.runid import generate_id
def _persist_settings(self, settings, settings_path) -> None:
    target_dir = os.path.dirname(settings_path)
    with tempfile.NamedTemporaryFile('w+', suffix='.tmp', delete=False, dir=target_dir) as fp:
        path = os.path.abspath(fp.name)
        with open(path, 'w+') as f:
            settings.write(f)
    try:
        os.replace(path, settings_path)
    except AttributeError:
        os.rename(path, settings_path)