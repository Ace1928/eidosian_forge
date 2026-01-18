import random
from typing import List, Tuple

import pygame as pg
from pygame.math import Vector2


class Apple:
    """
    A class representing the apple object in the game.
    """

    def __init__(self, grid_size: Tuple[int, int] = (40, 20)) -> None:
        """
        Initialize the Apple object.

        Args:
            grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (40, 20).
        """
        self.grid_size: Tuple[int, int] = grid_size
        self.boxes: List[Vector2] = [
            Vector2(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])
        ]
        self.position: Vector2 = self.generate(
            [Vector2(0, 0), Vector2(1, 0), Vector2(2, 0)]
        )

    def generate(self, snake_body: List[Vector2]) -> Vector2:
        """
        Generate a new position for the apple.

        Args:
            snake_body (List[Vector2]): The current positions of the snake body.

        Returns:
            Vector2: The new position of the apple, or None if there are no available positions.
        """
        empty_boxes: List[Vector2] = [
            box for box in self.boxes if box not in snake_body
        ]

        if not empty_boxes:
            return None

        self.position = random.choice(empty_boxes)
        return self.position

    def draw(self, screen: pg.Surface) -> None:
        """
        Draw the apple on the screen.

        Args:
            screen (pg.Surface): The surface to draw the apple on.
        """
        apple_rect: pg.Rect = pg.Rect(
            self.position.x * 30, self.position.y * 30, 30, 30
        )
        pg.draw.rect(screen, (255, 0, 0), apple_rect.inflate(-20, -20))
        pg.draw.rect(screen, (255, 0, 0), apple_rect.inflate(-10, -20))
        pg.draw.rect(screen, (255, 0, 0), apple_rect.inflate(-20, -10))
        pg.draw.rect(screen, (255, 0, 0), apple_rect.inflate(-10, -10))
