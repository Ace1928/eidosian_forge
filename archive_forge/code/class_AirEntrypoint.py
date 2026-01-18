import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
class AirEntrypoint(Enum):
    TUNER = 'Tuner.fit'
    TRAINER = 'Trainer.fit'
    TUNE_RUN = 'tune.run'
    TUNE_RUN_EXPERIMENTS = 'tune.run_experiments'