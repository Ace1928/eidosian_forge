from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class collide_rect_ratio:
    """A callable class that checks for collisions using scaled rects

    The class checks for collisions between two sprites using a scaled version
    of the sprites' rects. Is created with a ratio; the instance is then
    intended to be passed as a collided callback function to the *collide
    functions.

    New in pygame 1.8.1

    """

    def __init__(self, ratio):
        """create a new collide_rect_ratio callable

        Ratio is expected to be a floating point value used to scale
        the underlying sprite rect before checking for collisions.

        """
        self.ratio = ratio

    def __repr__(self):
        """
        Turn the class into a string.
        """
        return '<{klass} @{id:x} {attrs}>'.format(klass=self.__class__.__name__, id=id(self) & 16777215, attrs=' '.join((f'{k}={v!r}' for k, v in self.__dict__.items())))

    def __call__(self, left, right):
        """detect collision between two sprites using scaled rects

        pygame.sprite.collide_rect_ratio(ratio)(left, right): return bool

        Tests for collision between two sprites. Uses the pygame.Rect
        colliderect function to calculate the collision after scaling the rects
        by the stored ratio. Sprites must have "rect" attributes.

        """
        ratio = self.ratio
        leftrect = left.rect
        width = leftrect.width
        height = leftrect.height
        leftrect = leftrect.inflate(width * ratio - width, height * ratio - height)
        rightrect = right.rect
        width = rightrect.width
        height = rightrect.height
        rightrect = rightrect.inflate(width * ratio - width, height * ratio - height)
        return leftrect.colliderect(rightrect)