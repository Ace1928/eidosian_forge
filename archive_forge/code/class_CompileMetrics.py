import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
class CompileMetrics(metrics_module.Metric):

    def __init__(self, metrics, weighted_metrics, name='compile_metric', output_names=None):
        super().__init__(name=name)
        if metrics and (not isinstance(metrics, (list, tuple, dict))):
            raise ValueError(f'Expected `metrics` argument to be a list, tuple, or dict. Received instead: metrics={metrics} of type {type(metrics)}')
        if weighted_metrics and (not isinstance(weighted_metrics, (list, tuple, dict))):
            raise ValueError(f'Expected `weighted_metrics` argument to be a list, tuple, or dict. Received instead: weighted_metrics={weighted_metrics} of type {type(weighted_metrics)}')
        self._user_metrics = metrics
        self._user_weighted_metrics = weighted_metrics
        self.built = False
        self.name = 'compile_metrics'
        self.output_names = output_names

    @property
    def metrics(self):
        if not self.built:
            return []
        metrics = []
        for m in self._flat_metrics + self._flat_weighted_metrics:
            if isinstance(m, MetricsList):
                metrics.extend(m.metrics)
            elif m is not None:
                metrics.append(m)
        return metrics

    @property
    def variables(self):
        if not self.built:
            return []
        vars = []
        for m in self._flat_metrics + self._flat_weighted_metrics:
            if m is not None:
                vars.extend(m.variables)
        return vars

    def build(self, y_true, y_pred):
        if self.output_names:
            output_names = self.output_names
        elif isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all((hasattr(x, '_keras_history') for x in y_pred)):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
            num_outputs = 1
        if output_names:
            num_outputs = len(output_names)
        y_pred = self._flatten_y(y_pred)
        y_true = self._flatten_y(y_true)
        metrics = self._user_metrics
        weighted_metrics = self._user_weighted_metrics
        self._flat_metrics = self._build_metrics_set(metrics, num_outputs, output_names, y_true, y_pred, argument_name='metrics')
        self._flat_weighted_metrics = self._build_metrics_set(weighted_metrics, num_outputs, output_names, y_true, y_pred, argument_name='weighted_metrics')
        self.built = True

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

    def _flatten_y(self, y):
        if isinstance(y, dict) and self.output_names:
            result = []
            for name in self.output_names:
                if name in y:
                    result.append(y[name])
            return result
        return tree.flatten(y)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)
        y_true = self._flatten_y(y_true)
        y_pred = self._flatten_y(y_pred)
        for m, y_t, y_p in zip(self._flat_metrics, y_true, y_pred):
            if m:
                m.update_state(y_t, y_p)
        if sample_weight is not None:
            sample_weight = self._flatten_y(sample_weight)
            if len(sample_weight) < len(y_true):
                sample_weight = [sample_weight[0] for _ in range(len(y_true))]
        else:
            sample_weight = [None for _ in range(len(y_true))]
        for m, y_t, y_p, s_w in zip(self._flat_weighted_metrics, y_true, y_pred, sample_weight):
            if m:
                m.update_state(y_t, y_p, s_w)

    def reset_state(self):
        if not self.built:
            return
        for m in self._flat_metrics:
            if m:
                m.reset_state()
        for m in self._flat_weighted_metrics:
            if m:
                m.reset_state()

    def result(self):
        if not self.built:
            raise ValueError('Cannot get result() since the metric has not yet been built.')
        results = {}
        unique_name_counters = {}
        for mls in self._flat_metrics:
            if not mls:
                continue
            for m in mls.metrics:
                name = m.name
                if mls.output_name:
                    name = f'{mls.output_name}_{name}'
                if name not in unique_name_counters:
                    results[name] = m.result()
                    unique_name_counters[name] = 1
                else:
                    index = unique_name_counters[name]
                    unique_name_counters[name] += 1
                    name = f'{name}_{index}'
                    results[name] = m.result()
        for mls in self._flat_weighted_metrics:
            if not mls:
                continue
            for m in mls.metrics:
                name = m.name
                if mls.output_name:
                    name = f'{mls.output_name}_{name}'
                if name not in unique_name_counters:
                    results[name] = m.result()
                    unique_name_counters[name] = 1
                else:
                    name = f'weighted_{m.name}'
                    if mls.output_name:
                        name = f'{mls.output_name}_{name}'
                    if name not in unique_name_counters:
                        unique_name_counters[name] = 1
                    else:
                        index = unique_name_counters[name]
                        unique_name_counters[name] += 1
                        name = f'{name}_{index}'
                    results[name] = m.result()
        return results

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError