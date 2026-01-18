from __future__ import annotations
import datetime
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.layout.dimension import AnyDimension, D
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.utils import get_cwidth
class IterationsPerSecond(Formatter):
    """
    Display the iterations per second.
    """
    template = '<iterations-per-second>{iterations_per_second:.2f}</iterations-per-second>'

    def format(self, progress_bar: ProgressBar, progress: ProgressBarCounter[object], width: int) -> AnyFormattedText:
        value = progress.items_completed / progress.time_elapsed.total_seconds()
        return HTML(self.template.format(iterations_per_second=value))

    def get_width(self, progress_bar: ProgressBar) -> AnyDimension:
        all_values = [len(f'{c.items_completed / c.time_elapsed.total_seconds():.2f}') for c in progress_bar.counters]
        if all_values:
            return max(all_values)
        return 0