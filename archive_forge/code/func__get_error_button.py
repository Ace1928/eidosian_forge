from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
def _get_error_button(self, e):
    msg = str(e) if isinstance(e, PipelineError) else ''
    if self.debug:
        type, value, trb = sys.exc_info()
        tb_list = tb.format_tb(trb, None) + tb.format_exception_only(type, value)
        traceback = ('%s\n\nTraceback (innermost last):\n' + '%-20s %s') % (msg, ''.join(tb_list[-5:-1]), tb_list[-1])
    else:
        traceback = msg or 'Undefined error, enable debug mode.'
    button = Button(name='Error', button_type='danger', width=100, align='center', margin=(0, 0, 0, 5))
    button.js_on_click(code='alert(`{tb}`)'.format(tb=traceback))
    return button