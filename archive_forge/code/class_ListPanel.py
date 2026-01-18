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
class ListPanel(ListLike, Panel):
    """
    An abstract baseclass for Panel objects with list-like children.
    """
    scroll = param.Selector(default=False, objects=[False, True, 'both-auto', 'y-auto', 'x-auto', 'both', 'x', 'y'], doc='Whether to add scrollbars if the content overflows the size\n        of the container. If "both-auto", will only add scrollbars if\n        the content overflows in either directions. If "x-auto" or "y-auto",\n        will only add scrollbars if the content overflows in the\n        respective direction. If "both", will always add scrollbars.\n        If "x" or "y", will always add scrollbars in the respective\n        direction. If False, overflowing content will be clipped.\n        If True, will only add scrollbars in the direction of the container,\n        (e.g. Column: vertical, Row: horizontal).')
    _rename: ClassVar[Mapping[str, str | None]] = {'scroll': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'scroll': None}
    __abstract = True

    def __init__(self, *objects: Any, **params: Any):
        from ..pane import panel
        if objects:
            if 'objects' in params:
                raise ValueError("A %s's objects should be supplied either as positional arguments or as a keyword, not both." % type(self).__name__)
            params['objects'] = [panel(pane) for pane in objects]
        elif 'objects' in params:
            objects = params['objects']
            if not resolve_ref(objects) or iscoroutinefunction(objects):
                params['objects'] = [panel(pane) for pane in objects]
        super(Panel, self).__init__(**params)

    @property
    def _linked_properties(self):
        return tuple((self._property_mapping.get(p, p) for p in self.param if p not in ListPanel.param and self._property_mapping.get(p, p) is not None))

    def _process_param_change(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if (scroll := params.get('scroll')):
            css_classes = params.get('css_classes', self.css_classes)
            if scroll in _SCROLL_MAPPING:
                scroll_class = _SCROLL_MAPPING[scroll]
            elif self._direction:
                scroll_class = f'scrollable-{self._direction}'
            else:
                scroll_class = 'scrollable'
            params['css_classes'] = css_classes + [scroll_class]
        return super()._process_param_change(params)

    def _cleanup(self, root: Model | None=None) -> None:
        if root is not None and root.ref['id'] in state._fake_roots:
            state._fake_roots.remove(root.ref['id'])
        super()._cleanup(root)
        for p in self.objects:
            p._cleanup(root)