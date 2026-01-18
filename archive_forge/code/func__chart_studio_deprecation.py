import warnings
import functools
def _chart_studio_deprecation(fn):
    fn_name = fn.__name__
    fn_module = fn.__module__
    plotly_name = '.'.join(['plotly'] + fn_module.split('.')[1:] + [fn_name])
    chart_studio_name = '.'.join(['chart_studio'] + fn_module.split('.')[1:] + [fn_name])
    msg = '{plotly_name} is deprecated, please use {chart_studio_name}'.format(plotly_name=plotly_name, chart_studio_name=chart_studio_name)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return fn(*args, **kwargs)
    return wrapper