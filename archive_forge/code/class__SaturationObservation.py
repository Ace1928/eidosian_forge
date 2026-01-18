import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _SaturationObservation(LifeStatsObservation):
    """
    Returns the food saturation observation which determines how fast the hunger level depletes and is controlled by the
     kinds of food the player has eaten. Its maximum value always equals foodLevel's value and decreases with the hunger
     level. Its initial value on world creation is 5. - https://minecraft.wiki/w/Hunger#Mechanics
    """

    def __init__(self):
        super().__init__(hero_keys=['saturation'], univ_keys=['saturation'], space=spaces.Box(low=0, high=mc.MAX_FOOD_SATURATION, shape=(), dtype=float), default_if_missing=5.0)