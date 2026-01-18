from __future__ import annotations
import argparse
from ....commands.coverage import (
from ...environments import (
from .analyze import (
from .combine import (
from .erase import (
from .html import (
from .report import (
from .xml import (
def add_coverage_common(parser: argparse.ArgumentParser):
    """Add common coverage arguments."""
    parser.add_argument('--group-by', metavar='GROUP', action='append', choices=COVERAGE_GROUPS, help='group output by: %s' % ', '.join(COVERAGE_GROUPS))
    parser.add_argument('--all', action='store_true', help='include all python/powershell source files')
    parser.add_argument('--stub', action='store_true', help='generate empty report of all python/powershell source files')