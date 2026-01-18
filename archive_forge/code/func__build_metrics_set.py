import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
def _build_metrics_set(self, metrics, num_outputs, output_names, y_true, y_pred, argument_name):
    flat_metrics = []
    if isinstance(metrics, dict):
        for name in metrics.keys():
            if name not in output_names:
                raise ValueError(f"In the dict argument `{argument_name}`, key '{name}' does not correspond to any model output. Received:\n{argument_name}={metrics}")
    if num_outputs == 1:
        if not metrics:
            flat_metrics.append(None)
        else:
            if isinstance(metrics, dict):
                metrics = tree.flatten(metrics)
            if not isinstance(metrics, list):
                metrics = [metrics]
            if not all((is_function_like(m) for m in metrics)):
                raise ValueError(f'Expected all entries in the `{argument_name}` list to be metric objects. Received instead:\n{argument_name}={metrics}')
            flat_metrics.append(MetricsList([get_metric(m, y_true[0], y_pred[0]) for m in metrics if m is not None]))
    elif isinstance(metrics, (list, tuple)):
        if len(metrics) != len(y_pred):
            raise ValueError(f'For a model with multiple outputs, when providing the `{argument_name}` argument as a list, it should have as many entries as the model has outputs. Received:\n{argument_name}={metrics}\nof length {len(metrics)} whereas the model has {len(y_pred)} outputs.')
        for idx, (mls, yt, yp) in enumerate(zip(metrics, y_true, y_pred)):
            if not isinstance(mls, list):
                mls = [mls]
            name = output_names[idx] if output_names else None
            if not all((is_function_like(e) for e in mls)):
                raise ValueError(f'All entries in the sublists of the `{argument_name}` list should be metric objects. Found the following sublist with unknown types: {mls}')
            flat_metrics.append(MetricsList([get_metric(m, yt, yp) for m in mls if m is not None], output_name=name))
    elif isinstance(metrics, dict):
        if output_names is None:
            raise ValueError(f'Argument `{argument_name}` can only be provided as a dict when the model also returns a dict of outputs. Received {argument_name}={metrics}')
        for name in metrics.keys():
            if not isinstance(metrics[name], list):
                metrics[name] = [metrics[name]]
            if not all((is_function_like(e) for e in metrics[name])):
                raise ValueError(f"All entries in the sublists of the `{argument_name}` dict should be metric objects. At key '{name}', found the following sublist with unknown types: {metrics[name]}")
        for name, yt, yp in zip(output_names, y_true, y_pred):
            if name in metrics:
                flat_metrics.append(MetricsList([get_metric(m, yt, yp) for m in metrics[name] if m is not None], output_name=name))
            else:
                flat_metrics.append(None)
    return flat_metrics