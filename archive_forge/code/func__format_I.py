from reportlab.rl_config import register_reset
def _format_I(value):
    if value < 0 or value > 3999:
        raise ValueError('illegal value')
    str = ''
    base = -1
    while value:
        value, index = divmod(value, 10)
        tmp = _RN_TEMPLATES[index]
        while tmp:
            tmp, index = divmod(tmp, 8)
            str = _RN_LETTERS[index + base] + str
        base += 2
    return str