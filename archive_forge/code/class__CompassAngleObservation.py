import jinja2
import numpy as np
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
class _CompassAngleObservation(KeymapTranslationHandler):
    """
    Handles compass angle observations (converting to the correct angle offset normalized.)
    """

    def __init__(self):
        super().__init__(hero_keys=['compassAngle'], univ_keys=['compass', 'angle'], space=spaces.Box(low=-180.0, high=180.0, shape=(), dtype=np.float32), to_string='angle')

    def from_universal(self, obs):
        y = np.array((super().from_universal(obs) * 360.0 + 180) % 360.0 - 180)
        return y