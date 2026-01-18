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
def _get_max_version(distribution_files: List[Path]) -> Optional[str]:
    versions = []
    for p in distribution_files:
        version = _get_version_from_file(p)
        if version:
            try:
                versions.append(semantic_version.Version(version))
            except ValueError:
                return None
    return str(max(versions)) if versions else None