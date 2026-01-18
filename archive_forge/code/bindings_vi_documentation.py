from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import string
from prompt_toolkit.filters import IsReadOnly
from prompt_toolkit.key_binding.bindings.vi import create_operator_decorator
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
import six
A change operator.