from pathlib import Path
from wasabi import msg
from ..util import load_project_config, logger
from .main import Arg, app
from .remote_storage import RemoteStorage, get_command_hash, get_content_hash
def _is_not_empty_dir(loc: Path):
    if not loc.is_dir():
        return True
    elif any((_is_not_empty_dir(child) for child in loc.iterdir())):
        return True
    else:
        return False