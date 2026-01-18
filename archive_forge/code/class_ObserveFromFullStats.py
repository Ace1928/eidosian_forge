import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class ObserveFromFullStats(TranslationHandlerGroup):
    """
    Includes the use_item statistics for every item in MC that can be used
    """

    def xml_template(self) -> str:
        return str('<ObservationFromFullStats/>')

    def to_string(self) -> str:
        return self.stat_key

    def __init__(self, stat_key):
        if stat_key is None:
            self.stat_key = 'full_stats'
            super(ObserveFromFullStats, self).__init__(handlers=[_FullStatsObservation(statKeys) for statKeys in mc.ALL_STAT_KEYS if len(statKeys) == 2])
        else:
            self.stat_key = stat_key
            super(ObserveFromFullStats, self).__init__(handlers=[_FullStatsObservation(statKeys) for statKeys in mc.ALL_STAT_KEYS if stat_key in statKeys])