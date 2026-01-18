from __future__ import annotations
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import httpx
import semantic_version
from huggingface_hub import HfApi
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from tomlkit import parse
from typer import Argument, Option
from typing_extensions import Annotated
def _get_version_from_file(dist_file: Path) -> Optional[str]:
    match = re.search('-(\\d+\\.\\d+\\.\\d+[a-zA-Z]*\\d*)-', dist_file.name)
    if match:
        return match.group(1)