import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _FoodObservation(LifeStatsObservation):
    """
    Handles food_level observation representing the player's current hunger level, shown on the hunger bar. Its initial
    value on world creation is 20 (full bar) - https://minecraft.wiki/w/Hunger#Mechanics
    """

    def __init__(self):
        super().__init__(hero_keys=['food'], univ_keys=['food'], space=spaces.Box(low=0, high=mc.MAX_FOOD, shape=(), dtype=int), default_if_missing=mc.MAX_FOOD)