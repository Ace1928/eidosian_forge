import io
import sys
import typing
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from io import RawIOBase, UnsupportedOperation
from math import ceil
from mmap import mmap
from operator import length_hint
from os import PathLike, stat
from threading import Event, RLock, Thread
from types import TracebackType
from typing import (
from . import filesize, get_console
from .console import Console, Group, JustifyMethod, RenderableType
from .highlighter import Highlighter
from .jupyter import JupyterMixin
from .live import Live
from .progress_bar import ProgressBar
from .spinner import Spinner
from .style import StyleType
from .table import Column, Table
from .text import Text, TextType
class TaskProgressColumn(TextColumn):
    """Show task progress as a percentage.

    Args:
        text_format (str, optional): Format for percentage display. Defaults to "[progress.percentage]{task.percentage:>3.0f}%".
        text_format_no_percentage (str, optional): Format if percentage is unknown. Defaults to "".
        style (StyleType, optional): Style of output. Defaults to "none".
        justify (JustifyMethod, optional): Text justification. Defaults to "left".
        markup (bool, optional): Enable markup. Defaults to True.
        highlighter (Optional[Highlighter], optional): Highlighter to apply to output. Defaults to None.
        table_column (Optional[Column], optional): Table Column to use. Defaults to None.
        show_speed (bool, optional): Show speed if total is unknown. Defaults to False.
    """

    def __init__(self, text_format: str='[progress.percentage]{task.percentage:>3.0f}%', text_format_no_percentage: str='', style: StyleType='none', justify: JustifyMethod='left', markup: bool=True, highlighter: Optional[Highlighter]=None, table_column: Optional[Column]=None, show_speed: bool=False) -> None:
        self.text_format_no_percentage = text_format_no_percentage
        self.show_speed = show_speed
        super().__init__(text_format=text_format, style=style, justify=justify, markup=markup, highlighter=highlighter, table_column=table_column)

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        """Render the speed in iterations per second.

        Args:
            task (Task): A Task object.

        Returns:
            Text: Text object containing the task speed.
        """
        if speed is None:
            return Text('', style='progress.percentage')
        unit, suffix = filesize.pick_unit_and_suffix(int(speed), ['', '×10³', '×10⁶', '×10⁹', '×10¹²'], 1000)
        data_speed = speed / unit
        return Text(f'{data_speed:.1f}{suffix} it/s', style='progress.percentage')

    def render(self, task: 'Task') -> Text:
        if task.total is None and self.show_speed:
            return self.render_speed(task.finished_speed or task.speed)
        text_format = self.text_format_no_percentage if task.total is None else self.text_format
        _text = text_format.format(task=task)
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text