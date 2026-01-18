import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from parlai.core.opt import Opt
from parlai.core.teachers import (
from parlai.tasks.convai2.agents import (
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from .build import build
def _cached_data_path(opt: Opt, experiencer_side_only: bool) -> str:
    build(opt)
    dt = opt['datatype'].split(':')[0]
    side_string = 'experiencer_only' if experiencer_side_only else 'both_sides'
    return os.path.join(opt['datapath'], 'blended_skill_talk', f'ed_persona_topicifier__{dt}__{side_string}.json')