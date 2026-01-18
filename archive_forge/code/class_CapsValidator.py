import _plotly_utils.basevalidators
class CapsValidator(_plotly_utils.basevalidators.CompoundValidator):

    def __init__(self, plotly_name='caps', parent_name='volume', **kwargs):
        super(CapsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Caps'), data_docs=kwargs.pop('data_docs', '\n            x\n                :class:`plotly.graph_objects.volume.caps.X`\n                instance or dict with compatible properties\n            y\n                :class:`plotly.graph_objects.volume.caps.Y`\n                instance or dict with compatible properties\n            z\n                :class:`plotly.graph_objects.volume.caps.Z`\n                instance or dict with compatible properties\n'), **kwargs)