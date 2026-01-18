import os
import json
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
import parlai.tasks.google_sgd.build as build_
def _api_dict_to_str(self, apidict):
    return ' ; '.join((f'{k} = {v}' for k, v in apidict.items()))