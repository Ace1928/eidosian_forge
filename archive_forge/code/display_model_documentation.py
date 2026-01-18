from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
import random

Basic example which iterates through the tasks specified and runs the given model on
them.

Examples
--------

.. code-block:: shell

  parlai display_model -t babi:task1k:1 -m "repeat_label"
  parlai display_model -t "#MovieDD-Reddit" -m "ir_baseline" -mp "-lp 0.5" -dt test
