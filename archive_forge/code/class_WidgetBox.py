from __future__ import annotations
from collections import defaultdict, namedtuple
from typing import (
import param
from bokeh.models import Row as BkRow
from param.parameterized import iscoroutinefunction, resolve_ref
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import Column as PnColumn
from ..reactive import Reactive
from ..util import param_name, param_reprs, param_watchers
class WidgetBox(ListPanel):
    """
    The `WidgetBox` layout allows arranging multiple panel objects in a
    vertical (or horizontal) container.

    It is largely identical to the `Column` layout, but has some default
    styling that makes widgets be clearly grouped together visually.

    It has a list-like API with methods to `append`, `extend`, `clear`,
    `insert`, `pop`, `remove` and `__setitem__`, which make it possible to
    interactively update and modify the layout.

    Reference: https://panel.holoviz.org/reference/layouts/WidgetBox.html

    :Example:

    >>> pn.WidgetBox(some_widget, another_widget)
    """
    css_classes = param.List(default=['panel-widget-box'], doc='\n        CSS classes to apply to the layout.')
    disabled = param.Boolean(default=False, doc='\n        Whether the widget is disabled.')
    horizontal = param.Boolean(default=False, doc='\n        Whether to lay out the widgets in a Row layout as opposed\n        to a Column layout.')
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'disabled': None, 'horizontal': None}
    _rename: ClassVar[Mapping[str, str | None]] = {'disabled': None, 'objects': 'children', 'horizontal': None}
    _stylesheets: ClassVar[list[str]] = [f'{CDN_DIST}css/widgetbox.css', f'{CDN_DIST}css/listpanel.css']

    @property
    def _bokeh_model(self) -> Type[Model]:
        return BkRow if self.horizontal else PnColumn

    @property
    def _direction(self):
        return 'vertical' if self.horizontal else 'vertical'

    @param.depends('disabled', 'objects', watch=True)
    def _disable_widgets(self) -> None:
        from ..widgets import Widget
        for obj in self.select(Widget):
            obj.disabled = self.disabled

    def __init__(self, *objects: Any, **params: Any):
        super().__init__(*objects, **params)
        if self.disabled:
            self._disable_widgets()