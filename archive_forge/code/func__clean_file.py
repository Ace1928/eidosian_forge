import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def _clean_file(self, source: Path):
    """Clean an artifact."""
    artifacts, command = self._get_artifacts(source)
    for artifact in artifacts:
        remove_path(artifact)
        self._clean_file(artifact)