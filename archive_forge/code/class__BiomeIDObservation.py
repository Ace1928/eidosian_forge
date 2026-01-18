import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _BiomeIDObservation(_FullStatsObservation):

    def __init__(self):
        super().__init__(key_list=['biome_id'], space=spaces.Box(low=0, high=167, shape=(), dtype=int), default_if_missing=0)