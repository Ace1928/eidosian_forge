import collections
def _get_cartesian_label(x_or_y, r, c, cnt):
    label = '{x_or_y}{cnt}'.format(x_or_y=x_or_y, cnt=cnt)
    return label