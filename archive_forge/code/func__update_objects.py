from __future__ import annotations
from typing import ClassVar, List, Mapping
import param
from ..config import config
from ..io.resources import CDN_DIST, bundled_files
from ..reactive import ReactiveHTML
from ..util import classproperty
from .grid import GridSpec
@param.depends('state', watch=True)
def _update_objects(self):
    objects = {}
    object_ids = {str(id(obj)): obj for obj in self}
    for p in self.state:
        objects[p['y0'], p['x0'], p['y1'], p['x1']] = object_ids[p['id']]
    self.objects.clear()
    self.objects.update(objects)
    self._update_sizing()