import collections
def _get_cell_str(r, c, subplot_refs):
    layout_keys = sorted({k for ref in subplot_refs for k in ref.layout_keys})
    ref_str = ','.join(layout_keys)
    ref_str = ref_str.replace('axis', '')
    return '({r},{c}) {ref}'.format(r=r + 1, c=c + 1, ref=ref_str)