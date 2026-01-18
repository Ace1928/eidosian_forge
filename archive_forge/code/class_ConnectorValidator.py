import _plotly_utils.basevalidators
class ConnectorValidator(_plotly_utils.basevalidators.CompoundValidator):

    def __init__(self, plotly_name='connector', parent_name='waterfall', **kwargs):
        super(ConnectorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Connector'), data_docs=kwargs.pop('data_docs', '\n            line\n                :class:`plotly.graph_objects.waterfall.connecto\n                r.Line` instance or dict with compatible\n                properties\n            mode\n                Sets the shape of connector lines.\n            visible\n                Determines if connector lines are drawn.\n'), **kwargs)