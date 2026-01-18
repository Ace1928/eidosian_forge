import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _ScoreObservation(LifeStatsObservation):
    """
    Handles score observation
    """

    def __init__(self):
        keys = ['score']
        super().__init__(univ_keys=keys, hero_keys=keys, space=spaces.Box(low=0, high=mc.MAX_SCORE, shape=(), dtype=int), default_if_missing=0)