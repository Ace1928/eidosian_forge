import warnings
import functools
def _chart_studio_warning(submodule):
    warnings.warn('The plotly.{submodule} module is deprecated, please use chart_studio.{submodule} instead'.format(submodule=submodule), DeprecationWarning, stacklevel=2)