from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.utils.distributed import is_distributed
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import copy
import os
import tqdm

Generates a dictionary file from the training data.

Examples
--------

.. code-block:: shell

  # learn the vocabulary from one task, then train on another task.
  parlai build_dict -t convai2 --dict-file premade.dict
  parlai train_model -t squad --dict-file premade.dict -m seq2seq
