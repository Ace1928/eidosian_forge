from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils
def _save_annotation_body(self, cython_code, generated_code, annotation_items, scopes, covered_lines=None):
    outlist = [u'<div class="cython">']
    pos_comment_marker = u'/* … */\n'
    new_calls_map = dict(((name, 0) for name in 'refnanny trace py_macro_api py_c_api pyx_macro_api pyx_c_api error_goto'.split())).copy
    self.mark_pos(None)

    def annotate(match):
        group_name = match.lastgroup
        calls[group_name] += 1
        return u"<span class='%s'>%s</span>" % (group_name, match.group(group_name))
    lines = self._htmlify_code(cython_code, 'cython').splitlines()
    lineno_width = len(str(len(lines)))
    if not covered_lines:
        covered_lines = None
    for k, line in enumerate(lines, 1):
        try:
            c_code = generated_code[k]
        except KeyError:
            c_code = ''
        else:
            c_code = _replace_pos_comment(pos_comment_marker, c_code)
            if c_code.startswith(pos_comment_marker):
                c_code = c_code[len(pos_comment_marker):]
            c_code = html_escape(c_code)
        calls = new_calls_map()
        c_code = _parse_code(annotate, c_code)
        score = 5 * calls['py_c_api'] + 2 * calls['pyx_c_api'] + calls['py_macro_api'] + calls['pyx_macro_api']
        if c_code:
            onclick = self._onclick_attr
            expandsymbol = '+'
        else:
            onclick = ''
            expandsymbol = '&#xA0;'
        covered = ''
        if covered_lines is not None and k in covered_lines:
            hits = covered_lines[k]
            if hits is not None:
                covered = 'run' if hits else 'mis'
        outlist.append(u'<pre class="cython line score-{score}"{onclick}>{expandsymbol}<span class="{covered}">{line:0{lineno_width}d}</span>: {code}</pre>\n'.format(score=score, expandsymbol=expandsymbol, covered=covered, lineno_width=lineno_width, line=k, code=line.rstrip(), onclick=onclick))
        if c_code:
            outlist.append(u"<pre class='cython code score-{score} {covered}'>{code}</pre>".format(score=score, covered=covered, code=c_code))
    outlist.append(u'</div>')
    if self.show_entire_c_code:
        outlist.append(u'<p><div class="cython">')
        onclick_title = u"<pre class='cython line'{onclick}>+ {title}</pre>\n"
        outlist.append(onclick_title.format(onclick=self._onclick_attr, title=AnnotationCCodeWriter.COMPLETE_CODE_TITLE))
        complete_code_as_html = self._htmlify_code(self.buffer.getvalue(), 'c/cpp')
        outlist.append(u"<pre class='cython code'>{code}</pre>".format(code=complete_code_as_html))
        outlist.append(u'</div></p>')
    return outlist