from __future__ import annotations
from typing import (
import param
from ..io.resources import CDN_DIST
from .markup import Markdown

    The `Alert` pane allows providing contextual feedback messages for typical
    user actions. The Alert supports markdown strings.

    Reference: https://panel.holoviz.org/reference/panes/Alert.html

    :Example:

    >>> Alert('Some important message', alert_type='warning')
    