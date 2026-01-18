import collections
def _set_trace_grid_reference(trace, layout, grid_ref, row, col, secondary_y=False):
    if row <= 0:
        raise Exception('Row value is out of range. Note: the starting cell is (1, 1)')
    if col <= 0:
        raise Exception('Col value is out of range. Note: the starting cell is (1, 1)')
    try:
        subplot_refs = grid_ref[row - 1][col - 1]
    except IndexError:
        raise Exception('The (row, col) pair sent is out of range. Use Figure.print_grid to view the subplot grid. ')
    if not subplot_refs:
        raise ValueError('\nNo subplot specified at grid position ({row}, {col})'.format(row=row, col=col))
    if secondary_y:
        if len(subplot_refs) < 2:
            raise ValueError("\nSubplot with type '{subplot_type}' at grid position ({row}, {col}) was not\ncreated with the secondary_y spec property set to True. See the docstring\nfor the specs argument to plotly.subplots.make_subplots for more information.\n")
        trace_kwargs = subplot_refs[1].trace_kwargs
    else:
        trace_kwargs = subplot_refs[0].trace_kwargs
    for k in trace_kwargs:
        if k not in trace:
            raise ValueError("Trace type '{typ}' is not compatible with subplot type '{subplot_type}'\nat grid position ({row}, {col})\n\nSee the docstring for the specs argument to plotly.subplots.make_subplots\nfor more information on subplot types".format(typ=trace.type, subplot_type=subplot_refs[0].subplot_type, row=row, col=col))
    trace.update(trace_kwargs)