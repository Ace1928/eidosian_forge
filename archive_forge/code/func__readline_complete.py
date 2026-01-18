import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
def _readline_complete(self, text, state):
    context, prefix, except_last_word = self._analyze_tab_complete_input(text)
    candidates, _ = self._tab_completion_registry.get_completions(context, prefix)
    candidates = [except_last_word + candidate for candidate in candidates]
    return candidates[state]