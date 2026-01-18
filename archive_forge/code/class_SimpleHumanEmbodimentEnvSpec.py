import abc
from abc import ABC
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers as H, mc
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP, SIMPLE_KEYBOARD_ACTION
from minerl.herobraine.env_spec import EnvSpec
from typing import List
import numpy as np
class SimpleHumanEmbodimentEnvSpec(HumanControlEnvSpec):
    """
    A simpler base environment for legacy support of MineRL tasks.
    """

    def __init__(self, name, *args, resolution=(64, 64), **kwargs):
        self.resolution = resolution
        kwargs['resolution'] = resolution
        super().__init__(name, *args, **kwargs)

    def create_observables(self) -> List[TranslationHandler]:
        return [H.POVObservation(self.resolution)]

    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return [H.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items() if k in SIMPLE_KEYBOARD_ACTION] + [H.CameraAction()]

    def create_agent_start(self) -> List[Handler]:
        gui_handler = H.GuiScale(np.random.uniform(*self.guiscale_range))
        gamma_handler = H.GammaSetting(np.random.uniform(*self.gamma_range))
        fov_handler = H.FOVSetting(np.random.uniform(*self.fov_range))
        cursor_size_handler = H.FakeCursorSize(np.random.randint(self.cursor_size_range[0], self.cursor_size_range[1] + 1))
        return [gui_handler, gamma_handler, fov_handler, cursor_size_handler]

    def create_monitors(self) -> List[TranslationHandler]:
        return []