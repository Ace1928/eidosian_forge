from copy import deepcopy
def _is_storable(self, value):
    if not value:
        if not value in (0, 0.0, False):
            return False
    return True