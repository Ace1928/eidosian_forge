import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union
from .helpers import make_path
class MissingConfigItem(ConfigError):
    pass