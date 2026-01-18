import re
from string import Formatter
def _get_ansicode(self, tag):
    style = self._style
    foreground = self._foreground
    background = self._background
    if tag in style:
        return style[tag]
    elif tag in foreground:
        return foreground[tag]
    elif tag in background:
        return background[tag]
    elif tag.startswith('fg ') or tag.startswith('bg '):
        st, color = (tag[:2], tag[3:])
        code = '38' if st == 'fg' else '48'
        if st == 'fg' and color.lower() in foreground:
            return foreground[color.lower()]
        elif st == 'bg' and color.upper() in background:
            return background[color.upper()]
        elif color.isdigit() and int(color) <= 255:
            return '\x1b[%s;5;%sm' % (code, color)
        elif re.match('#(?:[a-fA-F0-9]{3}){1,2}$', color):
            hex_color = color[1:]
            if len(hex_color) == 3:
                hex_color *= 2
            rgb = tuple((int(hex_color[i:i + 2], 16) for i in (0, 2, 4)))
            return '\x1b[%s;2;%s;%s;%sm' % ((code,) + rgb)
        elif color.count(',') == 2:
            colors = tuple(color.split(','))
            if all((x.isdigit() and int(x) <= 255 for x in colors)):
                return '\x1b[%s;2;%s;%s;%sm' % ((code,) + colors)
    return None