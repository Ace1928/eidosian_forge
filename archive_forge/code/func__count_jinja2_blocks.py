from __future__ import (absolute_import, division, print_function)
def _count_jinja2_blocks(token, cur_depth, open_token, close_token):
    """
    this function counts the number of opening/closing blocks for a
    given opening/closing type and adjusts the current depth for that
    block based on the difference
    """
    num_open = token.count(open_token)
    num_close = token.count(close_token)
    if num_open != num_close:
        cur_depth += num_open - num_close
        if cur_depth < 0:
            cur_depth = 0
    return cur_depth