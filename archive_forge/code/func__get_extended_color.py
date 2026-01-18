import re
import markupsafe
def _get_extended_color(numbers):
    n = numbers.pop(0)
    if n == 2 and len(numbers) >= 3:
        r = numbers.pop(0)
        g = numbers.pop(0)
        b = numbers.pop(0)
        if not all((0 <= c <= 255 for c in (r, g, b))):
            raise ValueError()
    elif n == 5 and len(numbers) >= 1:
        idx = numbers.pop(0)
        if idx < 0:
            raise ValueError()
        if idx < 16:
            return idx
        if idx < 232:
            r = (idx - 16) // 36
            r = 55 + r * 40 if r > 0 else 0
            g = (idx - 16) % 36 // 6
            g = 55 + g * 40 if g > 0 else 0
            b = (idx - 16) % 6
            b = 55 + b * 40 if b > 0 else 0
        elif idx < 256:
            r = g = b = (idx - 232) * 10 + 8
        else:
            raise ValueError()
    else:
        raise ValueError()
    return (r, g, b)