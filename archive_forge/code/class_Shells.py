import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import click
class Shells(str, Enum):
    bash = 'bash'
    zsh = 'zsh'
    fish = 'fish'
    powershell = 'powershell'
    pwsh = 'pwsh'