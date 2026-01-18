from __future__ import annotations
from typing import (
import param
from ..models import Card as BkCard
from .base import Column, Row

    A `Card` layout allows arranging multiple panel objects in a
    collapsible, vertical container with a header bar.

    Reference: https://panel.holoviz.org/reference/layouts/Card.html

    :Example:

    >>> pn.Card(
    ...     some_widget, some_pane, some_python_object,
    ...     title='Card', styles=dict(background='WhiteSmoke'),
    ... )
    