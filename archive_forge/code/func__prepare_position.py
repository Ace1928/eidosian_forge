def _prepare_position(position, prepend_inside=False):
    if position is None:
        position = 'top right'
    pos_str = position
    position = set(position.split(' '))
    if prepend_inside:
        position = _add_inside_to_position(position)
    return (position, pos_str)