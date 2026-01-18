from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _facet_grid(df, x, y, facet_row, facet_col, num_of_rows, num_of_cols, facet_row_labels, facet_col_labels, trace_type, flipped_rows, flipped_cols, show_boxes, SUBPLOT_SPACING, marker_color, kwargs_trace, kwargs_marker):
    fig = make_subplots(rows=num_of_rows, cols=num_of_cols, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=SUBPLOT_SPACING, vertical_spacing=SUBPLOT_SPACING, print_grid=False)
    annotations = []
    if not facet_row and (not facet_col):
        trace = dict(type=trace_type, marker=dict(color=marker_color, line=kwargs_marker['line']), **kwargs_trace)
        if x:
            trace['x'] = df[x]
        if y:
            trace['y'] = df[y]
        trace = _make_trace_for_scatter(trace, trace_type, marker_color, **kwargs_marker)
        fig.append_trace(trace, 1, 1)
    elif facet_row and (not facet_col) or (not facet_row and facet_col):
        groups_by_facet = list(df.groupby(facet_row if facet_row else facet_col))
        for j, group in enumerate(groups_by_facet):
            trace = dict(type=trace_type, marker=dict(color=marker_color, line=kwargs_marker['line']), **kwargs_trace)
            if x:
                trace['x'] = group[1][x]
            if y:
                trace['y'] = group[1][y]
            trace = _make_trace_for_scatter(trace, trace_type, marker_color, **kwargs_marker)
            fig.append_trace(trace, j + 1 if facet_row else 1, 1 if facet_row else j + 1)
            label = _return_label(group[0], facet_row_labels if facet_row else facet_col_labels, facet_row if facet_row else facet_col)
            annotations.append(_annotation_dict(label, num_of_rows - j if facet_row else j + 1, num_of_rows if facet_row else num_of_cols, SUBPLOT_SPACING, 'row' if facet_row else 'col', flipped_rows))
    elif facet_row and facet_col:
        groups_by_facets = list(df.groupby([facet_row, facet_col]))
        tuple_to_facet_group = {item[0]: item[1] for item in groups_by_facets}
        row_values = df[facet_row].unique()
        col_values = df[facet_col].unique()
        for row_count, x_val in enumerate(row_values):
            for col_count, y_val in enumerate(col_values):
                try:
                    group = tuple_to_facet_group[x_val, y_val]
                except KeyError:
                    group = pd.DataFrame([[None, None]], columns=[x, y])
                trace = dict(type=trace_type, marker=dict(color=marker_color, line=kwargs_marker['line']), **kwargs_trace)
                if x:
                    trace['x'] = group[x]
                if y:
                    trace['y'] = group[y]
                trace = _make_trace_for_scatter(trace, trace_type, marker_color, **kwargs_marker)
                fig.append_trace(trace, row_count + 1, col_count + 1)
                if row_count == 0:
                    label = _return_label(col_values[col_count], facet_col_labels, facet_col)
                    annotations.append(_annotation_dict(label, col_count + 1, num_of_cols, SUBPLOT_SPACING, row_col='col', flipped=flipped_cols))
            label = _return_label(row_values[row_count], facet_row_labels, facet_row)
            annotations.append(_annotation_dict(label, num_of_rows - row_count, num_of_rows, SUBPLOT_SPACING, row_col='row', flipped=flipped_rows))
    return (fig, annotations)