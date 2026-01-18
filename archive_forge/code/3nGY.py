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


def main() -> None:
    """
    The main function to test and verify the functionality of the Apple class.
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
        snake_body: List[Vector2] = [Vector2(0, 0), Vector2(1, 0), Vector2(2, 0)]
        new_position: Vector2 = apple.generate(snake_body)
        assert new_position is not None, "Failed to generate a new apple position"
        assert (
            new_position not in snake_body
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
        snake_body_edge_case: List[Vector2] = [
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(2, 0),
        ]
        new_position_edge_case: Vector2 = apple_edge_case.generate(snake_body_edge_case)
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
