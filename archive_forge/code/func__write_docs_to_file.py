import itertools
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union
import srsly
from wasabi import Printer
from ..tokens import Doc, DocBin
from ..training import docs_to_json
from ..training.converters import (
from ._util import Arg, Opt, app, walk_directory
def _write_docs_to_file(data: Any, output_file: Path, output_type: str) -> None:
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    if output_type == 'json':
        srsly.write_json(output_file, data)
    else:
        with output_file.open('wb') as file_:
            file_.write(data)