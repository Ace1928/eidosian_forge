from .ipythonwidget import MetricVisualizer  # noqa
from .metrics_plotter import MetricsPlotter  # noqa
from .callbacks import XGBPlottingCallback, lgbm_plotting_callback  # noqa
def _jupyter_labextension_paths():
    return [{'src': 'labextension', 'dest': 'catboost-widget'}]