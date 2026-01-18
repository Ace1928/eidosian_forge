from . import errors
def _ensure_building(self):
    """Raise NotBuilding if there is no current tree being built."""
    if self._tree is None:
        raise NotBuilding