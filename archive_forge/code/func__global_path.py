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
def _global_path() -> Optional[str]:

    def try_create_dir(path) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            if os.access(path, os.W_OK):
                return True
        except OSError:
            pass
        return False

    def get_username() -> str:
        try:
            return getpass.getuser()
        except (ImportError, KeyError):
            return generate_id()
    try:
        home_config_dir = os.path.join(os.path.expanduser('~'), '.config', 'wandb')
        if not try_create_dir(home_config_dir):
            temp_config_dir = os.path.join(tempfile.gettempdir(), '.config', 'wandb')
            if not try_create_dir(temp_config_dir):
                username = get_username()
                config_dir = os.path.join(tempfile.gettempdir(), username, '.config', 'wandb')
                try_create_dir(config_dir)
            else:
                config_dir = temp_config_dir
        else:
            config_dir = home_config_dir
        config_dir = os.environ.get(env.CONFIG_DIR, config_dir)
        return os.path.join(config_dir, 'settings')
    except Exception:
        return None