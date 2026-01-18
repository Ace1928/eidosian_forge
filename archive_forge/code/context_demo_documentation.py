from __future__ import (absolute_import, division, print_function)
from ansible.plugins.callback import CallbackBase

    This is a very trivial example of how any callback function can get at play and task objects.
    play will be 'None' for runner invocations, and task will be None for 'setup' invocations.
    