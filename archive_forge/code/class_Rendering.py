import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math
from queue import PriorityQueue
class Rendering:
    """
    Handles all rendering operations for the game.
    """

    def __init__(self, window, snake: Snake, fruit: Fruit) -> None:
        self.window = window
        self.snake = snake
        self.fruit = fruit

    def render(self) -> None:
        """
        Renders all game objects.
        """
        self.window.fill(pg.Color(0, 0, 0))
        self.snake.draw()
        self.fruit.draw()
        self.draw_score()
        pg.display.flip()

    def draw_score(self) -> None:
        """
        Draws the current score on the screen.
        """
        font = pg.font.Font(None, 36)
        text = font.render(f'Score: {self.snake.score}', True, pg.Color(255, 255, 255))
        self.window.blit(text, (10, 10))