import json
from typing import List
import jinja2
from minerl.herobraine.hero.mc import EQUIPMENT_SLOTS
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler, TranslationHandlerGroup
import numpy as np
class _EquippedItemObservation(TranslationHandlerGroup):

    def to_string(self) -> str:
        return '_'.join([str(k) for k in self.keys])

    def __init__(self, dict_keys: List[str], items: List[str], _default, _other):
        self.keys = dict_keys
        super().__init__(handlers=[_TypeObservation(self.keys, items, _default=_default, _other=_other), _DamageObservation(self.keys, type_str='damage'), _DamageObservation(self.keys, type_str='maxDamage')])