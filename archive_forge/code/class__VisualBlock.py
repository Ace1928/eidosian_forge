import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
class _VisualBlock:
    """HTML Representation of Estimator

    Parameters
    ----------
    kind : {'serial', 'parallel', 'single'}
        kind of HTML block

    estimators : list of estimators or `_VisualBlock`s or a single estimator
        If kind != 'single', then `estimators` is a list of
        estimators.
        If kind == 'single', then `estimators` is a single estimator.

    names : list of str, default=None
        If kind != 'single', then `names` corresponds to estimators.
        If kind == 'single', then `names` is a single string corresponding to
        the single estimator.

    name_details : list of str, str, or None, default=None
        If kind != 'single', then `name_details` corresponds to `names`.
        If kind == 'single', then `name_details` is a single string
        corresponding to the single estimator.

    dash_wrapped : bool, default=True
        If true, wrapped HTML element will be wrapped with a dashed border.
        Only active when kind != 'single'.
    """

    def __init__(self, kind, estimators, *, names=None, name_details=None, dash_wrapped=True):
        self.kind = kind
        self.estimators = estimators
        self.dash_wrapped = dash_wrapped
        if self.kind in ('parallel', 'serial'):
            if names is None:
                names = (None,) * len(estimators)
            if name_details is None:
                name_details = (None,) * len(estimators)
        self.names = names
        self.name_details = name_details

    def _sk_visual_block_(self):
        return self