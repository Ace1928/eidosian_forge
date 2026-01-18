from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def _write_cell(self, s: Any, kind: str='td', indent: int=0, tags: str | None=None) -> None:
    if tags is not None:
        start_tag = f'<{kind} {tags}>'
    else:
        start_tag = f'<{kind}>'
    if self.escape:
        esc = {'&': '&amp;', '<': '&lt;', '>': '&gt;'}
    else:
        esc = {}
    rs = pprint_thing(s, escape_chars=esc).strip()
    if self.render_links and is_url(rs):
        rs_unescaped = pprint_thing(s, escape_chars={}).strip()
        start_tag += f'<a href="{rs_unescaped}" target="_blank">'
        end_a = '</a>'
    else:
        end_a = ''
    self.write(f'{start_tag}{rs}{end_a}</{kind}>', indent)