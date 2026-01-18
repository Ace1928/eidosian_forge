from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
import random
import tempfile

Build the candidate responses for a retrieval model.

Examples
--------

.. code-block:: shell

  parlai build_candidates -t convai2 --outfile /tmp/cands.txt
