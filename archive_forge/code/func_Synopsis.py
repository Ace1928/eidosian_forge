from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Synopsis(self, line, is_synopsis=False):
    """Renders NAME and SYNOPSIS lines as a hanging indent.

    Does not split top-level [...] or (...) groups.

    Args:
      line: The NAME or SYNOPSIS section text.
      is_synopsis: if it is the synopsis section
    """
    self._out.write('<dl class="notopmargin"><dt class="hangingindent"><span class="normalfont">\n')
    line = re.sub('(<code>)([-a-z0-9\\[\\]]+)(</code>)', '\\1<a href="#\\2">\\2</a>\\3', line)
    line = re.sub('href="#--no-', 'href="#--', line)
    line = re.sub('([^[=]\\[*<code><var>)([_A-Z0-9]+)(</var></code>)', '\\1<a href="#\\2">\\2</a>\\3', line)
    line = re.sub('(<code><a href="#[-a-z0-9\\[\\]]+">[-a-z0-9\\[\\]]+(<[^>]*>|\\S)*)', '<span class="flag">\\1</span>', line)
    if is_synopsis:
        line = re.sub('(,)', '\\1</span><span class="flag">', line)
    root = self.command[0].upper()
    line = re.sub('>{root}_WIDE_FLAG '.format(root=root), '><a href="#{root}-WIDE-FLAGS">{root}_WIDE_FLAG</a> '.format(root=root), line)
    nest = 0
    chars = collections.deque(line)
    while chars:
        c = chars.popleft()
        if c in '[(':
            nest += 1
            if nest == 1:
                c = '<span>' + c
        elif c in ')]':
            nest -= 1
            if not nest:
                c += '</span>'
        elif nest == 1 and c == ' ' and chars and (chars[0] == '|'):
            c = '</span> <span>&nbsp;&nbsp;&nbsp;&nbsp;' + chars.popleft()
        self._out.write(c)
    self._out.write('\n</span></dt></dl>\n')