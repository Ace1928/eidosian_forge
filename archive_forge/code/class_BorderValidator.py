import _plotly_utils.basevalidators
class BorderValidator(_plotly_utils.basevalidators.CompoundValidator):

    def __init__(self, plotly_name='border', parent_name='pointcloud.marker', **kwargs):
        super(BorderValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Border'), data_docs=kwargs.pop('data_docs', '\n            arearatio\n                Specifies what fraction of the marker area is\n                covered with the border.\n            color\n                Sets the stroke color. It accepts a specific\n                color. If the color is not fully opaque and\n                there are hundreds of thousands of points, it\n                may cause slower zooming and panning.\n'), **kwargs)