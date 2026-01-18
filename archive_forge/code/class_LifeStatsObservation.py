import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class LifeStatsObservation(KeymapTranslationHandler):

    def to_hero(self, x) -> str:
        pass

    def __init__(self, hero_keys, univ_keys, space, default_if_missing=None):
        self.hero_keys = hero_keys
        self.univ_keys = univ_keys
        super().__init__(hero_keys=hero_keys, univ_keys=['life_stats'] + univ_keys, space=space, default_if_missing=default_if_missing)

    def xml_template(self) -> str:
        return str('<ObservationFromFullStats/>')

    def from_hero(self, hero_dict):
        hero_dict = hero_dict['life_stats']
        return super().from_hero(hero_dict)