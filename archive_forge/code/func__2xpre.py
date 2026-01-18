import re
from io import StringIO
def _2xpre(s, styles):
    """Helper to transform Pygments HTML output to ReportLab markup"""
    s = s.replace('<div class="highlight">', '')
    s = s.replace('</div>', '')
    s = s.replace('<pre>', '')
    s = s.replace('</pre>', '')
    for k, c in styles + [('p', '#000000'), ('n', '#000000'), ('err', '#000000')]:
        s = s.replace('<span class="%s">' % k, '<span color="%s">' % c)
        s = re.sub('<span class="%s\\s+.*">' % k, '<span color="%s">' % c, s)
    s = re.sub('<span class=".*">', '<span color="#0f0f0f">', s)
    return s