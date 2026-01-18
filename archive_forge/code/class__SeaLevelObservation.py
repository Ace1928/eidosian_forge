import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _SeaLevelObservation(_FullStatsObservation):

    def __init__(self):
        super().__init__(key_list=['sea_level'], space=spaces.Box(low=0.0, high=255, shape=(), dtype=int), default_if_missing=63)