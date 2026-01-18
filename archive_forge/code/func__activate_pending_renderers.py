import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import (
from plotly.io._utils import validate_coerce_fig_to_dict
def _activate_pending_renderers(self, cls=object):
    """
        Activate all renderers that are waiting in the _to_activate list

        Parameters
        ----------
        cls
            Only activate renders that are subclasses of this class
        """
    to_activate_with_cls = [r for r in self._to_activate if cls and isinstance(r, cls)]
    while to_activate_with_cls:
        renderer = to_activate_with_cls.pop(0)
        renderer.activate()
    self._to_activate = [r for r in self._to_activate if not (cls and isinstance(r, cls))]