from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _add_shapes_to_fig(fig, annot_rect_color, flipped_rows=False, flipped_cols=False):
    shapes_list = []
    for key in fig['layout'].to_plotly_json().keys():
        if 'axis' in key and fig['layout'][key]['domain'] != [0.0, 1.0]:
            shape = {'fillcolor': annot_rect_color, 'layer': 'below', 'line': {'color': annot_rect_color, 'width': 1}, 'type': 'rect', 'xref': 'paper', 'yref': 'paper'}
            if 'xaxis' in key:
                shape['x0'] = fig['layout'][key]['domain'][0]
                shape['x1'] = fig['layout'][key]['domain'][1]
                shape['y0'] = 1.005
                shape['y1'] = 1.05
                if flipped_cols:
                    shape['y1'] += 0.5
                shapes_list.append(shape)
            elif 'yaxis' in key:
                shape['x0'] = 1.005
                shape['x1'] = 1.05
                shape['y0'] = fig['layout'][key]['domain'][0]
                shape['y1'] = fig['layout'][key]['domain'][1]
                if flipped_rows:
                    shape['x1'] += 1
                shapes_list.append(shape)
    fig['layout']['shapes'] = shapes_list