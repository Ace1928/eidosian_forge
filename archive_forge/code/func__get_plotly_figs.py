import json
import os
import warnings
from . import _catboost
def _get_plotly_figs(self, title):
    try:
        import plotly.graph_objs as go
    except ImportError as err:
        warnings.warn('To save plots to files you should install plotly.')
        raise ImportError(str(err))
    figs = []
    for path, dir_data in self.data.items():
        meta = dir_data['content']['data']['meta']
        has_parameters = meta['parameters'] == 'parameters'
        metrics = {}
        for i, learn_metric in enumerate(meta['learn_metrics']):
            metric_name = learn_metric['name']
            metrics.setdefault(metric_name, {})
            metrics[metric_name]['learn_sets'] = [(set_name, i) for set_name in meta['learn_sets']]
        for i, test_metric in enumerate(meta['test_metrics']):
            metric_name = test_metric['name']
            metrics.setdefault(metric_name, {})
            metrics[metric_name]['test_sets'] = [(set_name, i) for set_name in meta['test_sets']]
        iterations = dir_data['content']['data']['iterations']
        for metric_name, subsets in metrics.items():
            fig = go.Figure()
            figure_title = (title if dir_data['name'] == 'catboost_info' else dir_data['name']) + ' : ' + metric_name
            fig['layout']['title'] = go.layout.Title(text=figure_title)
            learn_graph_color = 'rgb(160,0,0)'
            for learn_set_name, metric_idx in subsets.get('learn_sets', []):
                fig.add_trace(go.Scatter(x=[e['iteration'] for e in iterations], y=[e[learn_set_name][metric_idx] for e in iterations], line=go.scatter.Line(color=learn_graph_color), mode='lines', name=learn_set_name))
            test_graph_color = 'rgb(0,160,0)'
            for test_set_name, metric_idx in subsets.get('test_sets', []):

                def generate_params_hover(e):
                    result = []
                    for param_name, param_value in e['parameters'][0].items():
                        result.append(param_name + ' : ' + str(param_value))
                    return '<br>'.join(result)
                fig.add_trace(go.Scatter(x=[e['iteration'] for e in iterations], y=[e[test_set_name][metric_idx] for e in iterations], line=go.scatter.Line(color=test_graph_color), mode='lines', name=test_set_name, hovertext=[generate_params_hover(e) for e in iterations] if has_parameters else None))
            fig.update_layout(xaxis=dict(title='iterations'), yaxis=dict(title=metric_name))
            figs.append(fig)
    return figs