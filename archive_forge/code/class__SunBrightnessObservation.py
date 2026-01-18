import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _SunBrightnessObservation(_FullStatsObservation):

    def __init__(self):
        super().__init__(key_list=['sun_brightness'], space=spaces.Box(low=0.0, high=1.0, shape=(), dtype=float), default_if_missing=0.94)