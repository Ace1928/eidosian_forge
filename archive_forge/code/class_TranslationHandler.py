from collections import OrderedDict
import logging
import numpy as np
from minerl.herobraine.hero.spaces import MineRLSpace
import minerl.herobraine.hero.spaces as spaces
from typing import List, Any
import typing
from minerl.herobraine.hero.handler import Handler
class TranslationHandler(Handler):
    """
    An agent handler to be added to the mission XML.
    This is useful as it defines basically all of the interfaces
    between universal action format, hero (malmo), and herobriane (ML stuff).
    """

    def __init__(self, space: MineRLSpace, **other_kwargs):
        self.space = space

    def from_hero(self, x: typing.Dict[str, Any]):
        """
        Converts a "hero" representation of an instance of this handler
        to a member of the space.
        """
        raise NotImplementedError()

    def to_hero(self, x) -> str:
        """
        Takes an instance of the handler, x in self.space, and maps it to
        the "hero" representation thereof.
        """
        raise NotImplementedError()

    def from_universal(self, x: typing.Dict[str, Any]):
        """sure
        Converts a universal representation of the handler (e.g. unviersal action/observation)
        """
        raise NotImplementedError()