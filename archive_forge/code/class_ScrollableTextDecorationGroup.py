from __future__ import annotations
from typing import Type, Optional, TYPE_CHECKING, Tuple
from pyglet import graphics
from pyglet.customtypes import AnchorY, AnchorX
from pyglet.gl import glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
from pyglet.text.layout.base import TextLayout
class ScrollableTextDecorationGroup(graphics.Group):
    scissor_area = (0, 0, 0, 0)

    def __init__(self, program: ShaderProgram, order: int=0, parent: Optional[graphics.Group]=None) -> None:
        """Create a text decoration rendering group.

        The group is created internally when a :py:class:`~pyglet.text.Label`
        is created; applications usually do not need to explicitly create it.
        """
        super().__init__(order=order, parent=parent)
        self.program = program

    def set_state(self) -> None:
        self.program.use()
        self.program['scissor'] = True
        self.program['scissor_area'] = self.scissor_area
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self) -> None:
        glDisable(GL_BLEND)
        self.program.stop()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(scissor={self.scissor_area})'

    def __eq__(self, other: graphics.Group) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)