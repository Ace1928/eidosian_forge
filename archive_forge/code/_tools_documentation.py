from __future__ import annotations
import logging # isort:skip
import itertools
import re
from dataclasses import dataclass
from typing import (
from ..models import (
from ..models.tools import (
from ..util.warnings import warn
 Adds tools to the plot object

    Args:
        plot (Plot): instance of a plot object

        tools (seq[Tool or str]|str): list of tool types or string listing the
            tool names. Those are converted using the to actual Tool instances.

        tooltips (string or seq[tuple[str, str]], optional):
            tooltips to use to configure a HoverTool

    Returns:
        list of Tools objects added to plot, map of supplied string names to tools
    