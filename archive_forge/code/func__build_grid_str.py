import collections
def _build_grid_str(specs, grid_ref, insets, insets_ref, row_seq):
    rows = len(specs)
    cols = len(specs[0])
    sp = '  '
    s_str = '[ '
    e_str = ' ]'
    s_top = '⎡ '
    s_mid = '⎢ '
    s_bot = '⎣ '
    e_top = ' ⎤'
    e_mid = ' ⎟'
    e_bot = ' ⎦'
    colspan_str = '       -'
    rowspan_str = '       :'
    empty_str = '    (empty) '
    grid_str = 'This is the format of your plot grid:\n'
    _tmp = [['' for c in range(cols)] for r in range(rows)]

    def _get_cell_str(r, c, subplot_refs):
        layout_keys = sorted({k for ref in subplot_refs for k in ref.layout_keys})
        ref_str = ','.join(layout_keys)
        ref_str = ref_str.replace('axis', '')
        return '({r},{c}) {ref}'.format(r=r + 1, c=c + 1, ref=ref_str)
    cell_len = max([len(_get_cell_str(r, c, ref)) for r, row_ref in enumerate(grid_ref) for c, ref in enumerate(row_ref) if ref]) + len(s_str) + len(e_str)

    def _pad(s, cell_len=cell_len):
        return ' ' * (cell_len - len(s))
    for r, spec_row in enumerate(specs):
        for c, spec in enumerate(spec_row):
            ref = grid_ref[r][c]
            if ref is None:
                if _tmp[r][c] == '':
                    _tmp[r][c] = empty_str + _pad(empty_str)
                continue
            if spec['rowspan'] > 1:
                cell_str = s_top + _get_cell_str(r, c, ref)
            else:
                cell_str = s_str + _get_cell_str(r, c, ref)
            if spec['colspan'] > 1:
                for cc in range(1, spec['colspan'] - 1):
                    _tmp[r][c + cc] = colspan_str + _pad(colspan_str)
                if spec['rowspan'] > 1:
                    _tmp[r][c + spec['colspan'] - 1] = colspan_str + _pad(colspan_str + e_str) + e_top
                else:
                    _tmp[r][c + spec['colspan'] - 1] = colspan_str + _pad(colspan_str + e_str) + e_str
            else:
                padding = ' ' * (cell_len - len(cell_str) - 2)
                if spec['rowspan'] > 1:
                    cell_str += padding + e_top
                else:
                    cell_str += padding + e_str
            if spec['rowspan'] > 1:
                for cc in range(spec['colspan']):
                    for rr in range(1, spec['rowspan']):
                        row_str = rowspan_str + _pad(rowspan_str)
                        if cc == 0:
                            if rr < spec['rowspan'] - 1:
                                row_str = s_mid + row_str[2:]
                            else:
                                row_str = s_bot + row_str[2:]
                        if cc == spec['colspan'] - 1:
                            if rr < spec['rowspan'] - 1:
                                row_str = row_str[:-2] + e_mid
                            else:
                                row_str = row_str[:-2] + e_bot
                        _tmp[r + rr][c + cc] = row_str
            _tmp[r][c] = cell_str + _pad(cell_str)
    for r in row_seq[::-1]:
        grid_str += sp.join(_tmp[r]) + '\n'
    if insets:
        grid_str += '\nWith insets:\n'
        for i_inset, inset in enumerate(insets):
            r = inset['cell'][0] - 1
            c = inset['cell'][1] - 1
            ref = grid_ref[r][c]
            subplot_labels_str = ','.join(insets_ref[i_inset][0].layout_keys)
            subplot_labels_str = subplot_labels_str.replace('axis', '')
            grid_str += s_str + subplot_labels_str + e_str + ' over ' + s_str + _get_cell_str(r, c, ref) + e_str + '\n'
    return grid_str