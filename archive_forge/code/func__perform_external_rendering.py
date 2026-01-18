import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import (
from plotly.io._utils import validate_coerce_fig_to_dict
def _perform_external_rendering(self, fig_dict, renderers_string=None, **kwargs):
    """
        Perform external rendering for each ExternalRenderer specified
        in either the default renderer string, or in the supplied
        renderers_string argument.

        Note that this method skips any renderers that are not subclasses
        of ExternalRenderer.

        Parameters
        ----------
        fig_dict: dict
            Figure dictionary
        renderers_string: str or None (default None)
            Renderer string to process rather than the current default
            renderer string

        Returns
        -------
        None
        """
    if renderers_string:
        renderer_names = self._validate_coerce_renderers(renderers_string)
        renderers_list = [self[name] for name in renderer_names]
        for renderer in renderers_list:
            if isinstance(renderer, ExternalRenderer):
                renderer.activate()
    else:
        self._activate_pending_renderers(cls=ExternalRenderer)
        renderers_list = self._default_renderers
    for renderer in renderers_list:
        if isinstance(renderer, ExternalRenderer):
            renderer = copy(renderer)
            for k, v in kwargs.items():
                if hasattr(renderer, k):
                    setattr(renderer, k, v)
            renderer.render(fig_dict)