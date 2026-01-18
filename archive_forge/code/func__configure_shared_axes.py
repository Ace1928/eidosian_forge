import collections
def _configure_shared_axes(layout, grid_ref, specs, x_or_y, shared, row_dir):
    rows = len(grid_ref)
    cols = len(grid_ref[0])
    layout_key_ind = ['x', 'y'].index(x_or_y)
    if row_dir < 0:
        rows_iter = range(rows - 1, -1, -1)
    else:
        rows_iter = range(rows)

    def update_axis_matches(first_axis_id, subplot_ref, spec, remove_label):
        if subplot_ref is None:
            return first_axis_id
        if x_or_y == 'x':
            span = spec['colspan']
        else:
            span = spec['rowspan']
        if subplot_ref.subplot_type == 'xy' and span == 1:
            if first_axis_id is None:
                first_axis_name = subplot_ref.layout_keys[layout_key_ind]
                first_axis_id = first_axis_name.replace('axis', '')
            else:
                axis_name = subplot_ref.layout_keys[layout_key_ind]
                axis_to_match = layout[axis_name]
                axis_to_match.matches = first_axis_id
                if remove_label:
                    axis_to_match.showticklabels = False
        return first_axis_id
    if shared == 'columns' or (x_or_y == 'x' and shared is True):
        for c in range(cols):
            first_axis_id = None
            ok_to_remove_label = x_or_y == 'x'
            for r in rows_iter:
                if not grid_ref[r][c]:
                    continue
                subplot_ref = grid_ref[r][c][0]
                spec = specs[r][c]
                first_axis_id = update_axis_matches(first_axis_id, subplot_ref, spec, ok_to_remove_label)
    elif shared == 'rows' or (x_or_y == 'y' and shared is True):
        for r in rows_iter:
            first_axis_id = None
            ok_to_remove_label = x_or_y == 'y'
            for c in range(cols):
                if not grid_ref[r][c]:
                    continue
                subplot_ref = grid_ref[r][c][0]
                spec = specs[r][c]
                first_axis_id = update_axis_matches(first_axis_id, subplot_ref, spec, ok_to_remove_label)
    elif shared == 'all':
        first_axis_id = None
        for c in range(cols):
            for ri, r in enumerate(rows_iter):
                if not grid_ref[r][c]:
                    continue
                subplot_ref = grid_ref[r][c][0]
                spec = specs[r][c]
                if x_or_y == 'y':
                    ok_to_remove_label = c > 0
                else:
                    ok_to_remove_label = ri > 0 if row_dir > 0 else r < rows - 1
                first_axis_id = update_axis_matches(first_axis_id, subplot_ref, spec, ok_to_remove_label)