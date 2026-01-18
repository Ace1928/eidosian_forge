from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
import random
import tempfile
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script

Convert a dataset into the ParlAI text format.

Examples
--------

.. code-block:: shell

  parlai convert_data_to_fasttext_format -t babi:task1k:1 --outfile /tmp/dump
