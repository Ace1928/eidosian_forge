from __future__ import annotations
import datetime
import json
import sys
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage import __version__
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf, TLineNo
def _convert_branch_arcs(branch_arcs: dict[TLineNo, list[TLineNo]]) -> Iterable[tuple[TLineNo, TLineNo]]:
    """Convert branch arcs to a list of two-element tuples."""
    for source, targets in branch_arcs.items():
        for target in targets:
            yield (source, target)