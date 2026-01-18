from ipywidgets import register
from .Geometry_autogen import Geometry as AutogenGeometry
from .._base.Three import ThreeWidget
def _repr_keys(self):
    return filter(_make_key_filter(self._store_ref), super(Geometry, self)._repr_keys())