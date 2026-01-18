import os
import pygame as pg
def DisplayGradient(surf):
    """choose random colors and show them"""
    stopwatch()
    colors = np_random.randint(0, 255, (2, 3))
    column = VertGradientColumn(surf, colors[0], colors[1])
    pg.surfarray.blit_array(surf, column)
    pg.display.flip()
    stopwatch('Gradient:')