import os
import sys
from typing import (  # noqa: F401
@runtime_checkable
class Has_Repo(Protocol):
    repo: 'Repo'