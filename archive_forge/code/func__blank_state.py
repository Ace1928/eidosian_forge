import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def _blank_state(self, js_source_text=None):
    if js_source_text is None:
        js_source_text = ''
    self._flags = None
    self._previous_flags = None
    self._flag_store = []
    self._tokens = None
    if self._options.eol == 'auto':
        self._options.eol = '\n'
        if self.acorn.lineBreak.search(js_source_text or ''):
            self._options.eol = self.acorn.lineBreak.search(js_source_text).group()
    baseIndentString = re.search('^[\t ]*', js_source_text).group(0)
    self._last_last_text = ''
    self._output = Output(self._options, baseIndentString)
    self._output.raw = self._options.test_output_raw
    self.set_mode(MODE.BlockStatement)
    return js_source_text