import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _YawObservation(_FullStatsObservation):

    def __init__(self):
        super().__init__(key_list=['yaw'], space=spaces.Box(low=-180.0, high=180.0, shape=(), dtype=float), default_if_missing=0.0)