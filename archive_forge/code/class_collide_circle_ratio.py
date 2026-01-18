from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class collide_circle_ratio:
    """detect collision between two sprites using scaled circles

    This callable class checks for collisions between two sprites using a
    scaled version of a sprite's radius. It is created with a ratio as the
    argument to the constructor. The instance is then intended to be passed as
    a collided callback function to the *collide functions.

    New in pygame 1.8.1

    """

    def __init__(self, ratio):
        """creates a new collide_circle_ratio callable instance

        The given ratio is expected to be a floating point value used to scale
        the underlying sprite radius before checking for collisions.

        When the ratio is ratio=1.0, then it behaves exactly like the
        collide_circle method.

        """
        self.ratio = ratio

    def __repr__(self):
        """
        Turn the class into a string.
        """
        return '<{klass} @{id:x} {attrs}>'.format(klass=self.__class__.__name__, id=id(self) & 16777215, attrs=' '.join((f'{k}={v!r}' for k, v in self.__dict__.items())))

    def __call__(self, left, right):
        """detect collision between two sprites using scaled circles

        pygame.sprite.collide_circle_radio(ratio)(left, right): return bool

        Tests for collision between two sprites by testing whether two circles
        centered on the sprites overlap after scaling the circle's radius by
        the stored ratio. If the sprites have a "radius" attribute, that is
        used to create the circle; otherwise, a circle is created that is big
        enough to completely enclose the sprite's rect as given by the "rect"
        attribute. Intended to be passed as a collided callback function to the
        *collide functions. Sprites must have a "rect" and an optional "radius"
        attribute.

        """
        ratio = self.ratio
        xdistance = left.rect.centerx - right.rect.centerx
        ydistance = left.rect.centery - right.rect.centery
        distancesquared = xdistance ** 2 + ydistance ** 2
        try:
            leftradius = left.radius
        except AttributeError:
            leftrect = left.rect
            leftradius = 0.5 * (leftrect.width ** 2 + leftrect.height ** 2) ** 0.5
            left.radius = leftradius
        leftradius *= ratio
        try:
            rightradius = right.radius
        except AttributeError:
            rightrect = right.rect
            rightradius = 0.5 * (rightrect.width ** 2 + rightrect.height ** 2) ** 0.5
            right.radius = rightradius
        rightradius *= ratio
        return distancesquared <= (leftradius + rightradius) ** 2