import os
import random
from typing import List
import pygame as pg
class Bomb(pg.sprite.Sprite):
    """A bomb the aliens drop."""
    speed = 9
    images: List[pg.Surface] = []

    def __init__(self, alien, explosion_group, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=alien.rect.move(0, 5).midbottom)
        self.explosion_group = explosion_group

    def update(self):
        """called every time around the game loop.

        Every frame we move the sprite 'rect' down.
        When it reaches the bottom we:

        - make an explosion.
        - remove the Bomb.
        """
        self.rect.move_ip(0, self.speed)
        if self.rect.bottom >= 470:
            Explosion(self, self.explosion_group)
            self.kill()