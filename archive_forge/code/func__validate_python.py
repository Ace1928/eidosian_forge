import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def _validate_python(self, value, state):
    value = str(value).strip().upper()
    if not value:
        raise Invalid(self.message('empty', state), value, state)
    if not value or len(value) != 2:
        raise Invalid(self.message('wrongLength', state), value, state)
    if value not in self.states and (not (self.extra_states and value in self.extra_states)):
        raise Invalid(self.message('invalid', state), value, state)