from __future__ import annotations
import logging # isort:skip
import inspect
import os
import warnings  # lgtm [py/import-and-import-from]
class BokehDeprecationWarning(DeprecationWarning):
    """ A Bokeh-specific ``DeprecationWarning`` subclass.

    Used to selectively filter Bokeh deprecations for unconditional display.

    """