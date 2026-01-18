from __future__ import annotations
import logging # isort:skip
import inspect
import os
import warnings  # lgtm [py/import-and-import-from]
class BokehUserWarning(UserWarning):
    """ A Bokeh-specific ``UserWarning`` subclass.

    Used to selectively filter Bokeh warnings for unconditional display.

    """