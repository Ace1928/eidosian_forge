import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _BreathObservation(LifeStatsObservation):
    """
        Handles observation of breath which tracks the amount of air remaining before beginning to suffocate
    """

    def __init__(self):
        super().__init__(hero_keys=['air'], univ_keys=['air'], space=spaces.Box(low=0, high=mc.MAX_BREATH, shape=(), dtype=int), default_if_missing=300)