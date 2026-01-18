import sys
import os
import pygame as pg
from pygame.transform import scale
def add_arrow_button(screen, regions, posn, direction):
    draw_arrow(screen, 'black', posn, direction)
    draw_arrow(regions, (direction, 0, 0), posn, direction)