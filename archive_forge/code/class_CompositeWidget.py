from __future__ import annotations
import math
from typing import (
import param  # type: ignore
from bokeh.models import ImportedStyleSheet, Tooltip
from bokeh.models.dom import HTML
from param.parameterized import register_reference_transform
from .._param import Margin
from ..layout.base import Row
from ..reactive import Reactive
from ..viewable import Layoutable, Viewable
class CompositeWidget(Widget):
    """
    A baseclass for widgets which are made up of two or more other
    widgets
    """
    _composite_type: ClassVar[Type[ListPanel]] = Row
    _linked_properties: ClassVar[Tuple[str]] = ()
    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        layout_params = [p for p in Layoutable.param if p != 'name']
        layout = {p: getattr(self, p) for p in layout_params if getattr(self, p) is not None}
        if layout.get('width', self.width) is None and 'sizing_mode' not in layout:
            layout['sizing_mode'] = 'stretch_width'
        if layout.get('sizing_mode') not in (None, 'fixed') and layout.get('width'):
            min_width = layout.pop('width')
            if not layout.get('min_width'):
                layout['min_width'] = min_width
        self._composite = self._composite_type(**layout)
        self._models = self._composite._models
        self._internal_callbacks.append(self.param.watch(self._update_layout_params, layout_params))

    def _update_layout_params(self, *events: param.parameterized.Event) -> None:
        updates = {event.name: event.new for event in events}
        self._composite.param.update(**updates)

    def select(self, selector: Optional[type | Callable[['Viewable'], bool]]=None) -> List[Viewable]:
        """
        Iterates over the Viewable and any potential children in the
        applying the Selector.

        Arguments
        ---------
        selector: type or callable or None
          The selector allows selecting a subset of Viewables by
          declaring a type or callable function to filter by.

        Returns
        -------
        viewables: list(Viewable)
        """
        objects = super().select(selector)
        for obj in self._composite.objects:
            objects += obj.select(selector)
        return objects

    def _cleanup(self, root: Model | None=None) -> None:
        self._composite._cleanup(root)
        super()._cleanup(root)

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        model = self._composite._get_model(doc, root, parent, comm)
        root = root or model
        self._models[root.ref['id']] = (model, parent)
        return model

    def __contains__(self, object: Any) -> bool:
        return object in self._composite.objects

    @property
    def _synced_params(self) -> List[str]:
        return []