import os
import random
from typing import List
import pygame as pg
class Score(pg.sprite.Sprite):
    """to keep track of the score."""

    def __init__(self, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.font = pg.font.Font(None, 20)
        self.font.set_italic(1)
        self.color = 'white'
        self.lastscore = -1
        self.update()
        self.rect = self.image.get_rect().move(10, 450)

    def update(self):
        """We only update the score in update() when it has changed."""
        if SCORE != self.lastscore:
            self.lastscore = SCORE
            msg = f'Score: {SCORE}'
            self.image = self.font.render(msg, 0, self.color)