import random
from typing import List, Optional, Tuple

import pygame as pg
from pygame.math import Vector2
import numpy as np


class Apple:
    """
    A class representing the apple object in the game, optimized with advanced data structures and algorithms.
    """

    def __init__(self, grid_size: Tuple[int, int] = (40, 20)) -> None:
        """
        Initialize the Apple object with a grid size, precomputing all possible positions for efficiency.

        Args:
            grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (40, 20).
        """
        self.grid_size: Tuple[int, int] = grid_size
        self.boxes: np.ndarray = np.array(
            [[i, j] for i in range(grid_size[0]) for j in range(grid_size[1])],
            dtype=np.int32,
        )
        self.position: np.ndarray = self.generate(
            np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int32)
        )

    def generate(self, snake_body: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a new position for the apple using optimized numpy operations for maximum efficiency.

        Args:
            snake_body (np.ndarray): The current positions of the snake body.

        Returns:
            Optional[np.ndarray]: The new position of the apple, or None if there are no available positions.
        """
        empty_boxes: np.ndarray = self.boxes[
            ~np.any(
                np.isin(self.boxes, snake_body).reshape(self.boxes.shape[0], -1), axis=1
            )
        ]

        if empty_boxes.size == 0:
            return None

        self.position = empty_boxes[np.random.choice(empty_boxes.shape[0])]
        return self.position

    def draw(self, screen: pg.Surface) -> None:
        """
        Draw the apple on the screen using optimized Pygame methods for enhanced performance.

        Args:
            screen (pg.Surface): The surface to draw the apple on.
        """
        apple_rect: pg.Rect = pg.Rect(
            int(self.position[0] * 30), int(self.position[1] * 30), 30, 30
        )
        apple_surface: pg.Surface = pg.Surface((30, 30), flags=pg.SRCALPHA)
        apple_surface.fill((0, 0, 0, 0))
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-20, -20),
        )
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-10, -20),
        )
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-20, -10),
        )
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-10, -10),
        )
        screen.blit(apple_surface, apple_rect)


def main() -> None:
    """
    The main function to test and verify the functionality of the Apple class with comprehensive testing and validation.
    """
    try:
        # Initialize Pygame
        pg.init()
        screen: pg.Surface = pg.display.set_mode((1200, 600))
        pg.display.set_caption("Apple Class Test")
        clock: pg.time.Clock = pg.time.Clock()

        # Create an instance of the Apple class
        apple: Apple = Apple()

        # Test the generate method
        snake_body: np.ndarray = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int32)
        new_position: Optional[np.ndarray] = apple.generate(snake_body)
        assert new_position is not None, "Failed to generate a new apple position"
        assert not np.any(
            np.all(new_position == snake_body, axis=1)
        ), "Generated apple position overlaps with snake body"

        # Test the draw method
        apple.draw(screen)
        pg.display.flip()
        pg.time.delay(
            1000
        )  # Pause for 1 second to visually verify the apple is drawn correctly

        # Test edge case: generate apple when no empty positions are available
        grid_size: Tuple[int, int] = (3, 1)
        apple_edge_case: Apple = Apple(grid_size)
        snake_body_edge_case: np.ndarray = np.array(
            [[0, 0], [1, 0], [2, 0]], dtype=np.int32
        )
        new_position_edge_case: Optional[np.ndarray] = apple_edge_case.generate(
            snake_body_edge_case
        )
        assert (
            new_position_edge_case is None
        ), "Generated apple position when no empty positions available"

        print("All tests passed successfully!")

    except AssertionError as e:
        print(f"Test failed: {str(e)}")

    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")

    finally:
        pg.quit()


if __name__ == "__main__":
    main()
