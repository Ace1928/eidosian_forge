import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
def _get_visual_block(estimator):
    """Generate information about how to display an estimator."""
    if hasattr(estimator, '_sk_visual_block_'):
        try:
            return estimator._sk_visual_block_()
        except Exception:
            return _VisualBlock('single', estimator, names=estimator.__class__.__name__, name_details=str(estimator))
    if isinstance(estimator, str):
        return _VisualBlock('single', estimator, names=estimator, name_details=estimator)
    elif estimator is None:
        return _VisualBlock('single', estimator, names='None', name_details='None')
    if hasattr(estimator, 'get_params') and (not isclass(estimator)):
        estimators = [(key, est) for key, est in estimator.get_params(deep=False).items() if hasattr(est, 'get_params') and hasattr(est, 'fit') and (not isclass(est))]
        if estimators:
            return _VisualBlock('parallel', [est for _, est in estimators], names=[f'{key}: {est.__class__.__name__}' for key, est in estimators], name_details=[str(est) for _, est in estimators])
    return _VisualBlock('single', estimator, names=estimator.__class__.__name__, name_details=str(estimator))