import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _XPObservation(LifeStatsObservation):
    """
    Handles observation of experience points 1395 experience corresponds with level 30
    - see https://minecraft.wiki/w/Experience for more details
    """

    def __init__(self):
        keys = ['xp']
        super().__init__(hero_keys=keys, univ_keys=keys, space=spaces.Box(low=0, high=mc.MAX_XP, shape=(), dtype=int), default_if_missing=0)