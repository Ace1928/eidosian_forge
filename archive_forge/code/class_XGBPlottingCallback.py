from IPython.display import display
from .metrics_plotter import MetricsPlotter
class XGBPlottingCallback(XGBTrainingCallback):
    """XGBoost callback with metrics plotting widget from CatBoost
    """

    def __init__(self, total_iterations: int):
        self.plotter = None
        self.total_iterations = total_iterations

    def after_iteration(self, model, epoch, evals_log):
        data_names = evals_log.keys()
        first_train = all(['valid' in data_name.lower() for data_name in data_names]) and len(data_names) > 1
        for data_name, metrics_info in evals_log.items():
            if 'train' in data_name.lower() or first_train:
                train = True
                first_train = False
            elif 'valid' in data_name.lower() or 'test' in data_name.lower():
                train = False
            else:
                raise Exception('Unexpected sample name during evaluation')
            metrics = {name: values[-1] for name, values in metrics_info.items()}
            if self.plotter is None:
                names = list(metrics.keys())
                self.plotter = MetricsPlotter(names, names, self.total_iterations)
                display(self.plotter._widget)
            self.plotter.log(epoch, train, metrics)
        return False