from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
class Apple:
    """
    A class representing the apple object in the game, optimized with advanced data structures and algorithms.
    """

    def __init__(self, grid_size: Tuple[int, int]=(40, 20)) -> None:
        """
        Initialize the Apple object with a grid size, precomputing all possible positions for efficiency.

        Args:
            grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (40, 20).
        """
        self.grid_size: Tuple[int, int] = grid_size
        self.boxes: np.ndarray = np.array([[i, j] for i in range(grid_size[0]) for j in range(grid_size[1])], dtype=np.int32)
        self.position: np.ndarray = self.generate(np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int32))

    def generate(self, snake_body: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a new position for the apple using optimized numpy operations for maximum efficiency.

        Args:
            snake_body (np.ndarray): The current positions of the snake body.

        Returns:
            Optional[np.ndarray]: The new position of the apple, or None if there are no available positions.
        """
        empty_boxes: np.ndarray = self.boxes[~np.any(np.isin(self.boxes, snake_body).reshape(self.boxes.shape[0], -1), axis=1)]
        if empty_boxes.size == 0:
            return None
        self.position = empty_boxes[np.random.choice(empty_boxes.shape[0])]
        return self.position

    def show(self, screen: pg.Surface) -> None:
        """
        Draw the apple on the screen using optimized Pygame methods for enhanced performance.

        Args:
            screen (pg.Surface): The surface to draw the apple on.
        """
        apple_rect: pg.Rect = pg.Rect(int(self.position[0] * 30), int(self.position[1] * 30), 30, 30)
        apple_surface: pg.Surface = pg.Surface((30, 30), flags=pg.SRCALPHA)
        apple_surface.fill((0, 0, 0, 0))
        pg.draw.rect(apple_surface, pg.Color(255, 0, 0), apple_surface.get_rect().inflate(-20, -20))
        pg.draw.rect(apple_surface, pg.Color(255, 0, 0), apple_surface.get_rect().inflate(-10, -20))
        pg.draw.rect(apple_surface, pg.Color(255, 0, 0), apple_surface.get_rect().inflate(-20, -10))
        pg.draw.rect(apple_surface, pg.Color(255, 0, 0), apple_surface.get_rect().inflate(-10, -10))
        screen.blit(apple_surface, apple_rect)