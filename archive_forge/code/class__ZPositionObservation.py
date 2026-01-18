import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _ZPositionObservation(_FullStatsObservation):

    def __init__(self):
        super().__init__(key_list=['zpos'], space=spaces.Box(low=-640000.0, high=640000.0, shape=(), dtype=float), default_if_missing=0.0)