# // Creating the snake, apple and A* search algorithm
# Creating the snake, apple and search algorithm

import apple
from search import SearchAlgorithm

from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint

# Initialize Pygame
pg.init()
# Initialize the display
pg.display.init()
# Retrieve the current display information
display_info = pg.display.Info()


def calculate_block_size(screen_width: int, screen_height: int) -> int:
    reference_resolution = (1920, 1080)
    reference_block_size = 20

    scaling_factor_width = screen_width / reference_resolution[0]
    scaling_factor_height = screen_height / reference_resolution[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


block_size = calculate_block_size(display_info.current_w, display_info.current_h)
border_width = 3 * block_size
screen_size = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)
border_color = (255, 255, 255)
apple_instance = apple.Apple()
clock = pg.time.Clock()
frames_per_second = 60
tick_rate = 1000 // frames_per_second


def setup_environment() -> (
    Tuple[pg.Surface, apple.Apple, SearchAlgorithm, pg.time.Clock]
):
    pg.init()
    screen: pg.Surface = pg.display.set_mode(screen_size)
    apple_object: apple.Apple = apple_instance
    search_algorithm: SearchAlgorithm = SearchAlgorithm()
    search_algorithm.get_path()
    clock: pg.time.Clock = clock
    return screen, apple_object, search_algorithm, clock


class Snake:
    def __init__(self):
        self.body: List[Vector2] = [Vector2(i, 0) for i in range(3)]
        self.x_direction: int = 1
        self.y_direction: int = 0
        self.path: List[Vector2] = []
        self.tail_position: Vector2 = self.get_tail_position()

    def update(self, apple: apple.Apple) -> None:
        if (
            self.get_head_position().x == self.tail_position.x
            and self.get_head_position().y == self.tail_position.y
        ) or len(self.body) > 600:
            self.tail_position = Vector2(0, 0)
            self.tail_position.x = self.get_tail_position().x
            self.tail_position.y = self.get_tail_position().y
            SearchAlgorithm.get_path()

        for i in range(len(self.path)):
            head: Vector2 = self.get_head_position()
            if head.x == self.path[i].x and head.y == self.path[i].y:
                next_head: Vector2 = self.path[i + 1]
                if next_head.x - head.x == 1:
                    self.move_right()
                elif next_head.x - head.x == -1:
                    self.move_left()
                elif next_head.y - head.y == 1:
                    self.move_down()
                elif next_head.y - head.y == -1:
                    self.move_up()
                else:
                    print("Something is wrong")

        if self.get_head_position().x == 39 and self.x_direction == 1:
            print("Collision with wall")
        elif self.get_head_position().x == 0 and self.x_direction == -1:
            print("Collision with wall")
        elif self.get_head_position().y == 19 and self.y_direction == 1:
            print("Collision with wall")
        elif self.get_head_position().y == 0 and self.y_direction == -1:
            print("Collision with wall")
        else:
            for i in range(len(self.body) - 1):
                if (
                    (
                        self.get_head_position().x == self.body[i].x
                        and self.get_head_position().y - self.body[i].y == 1
                        and self.y_direction == -1
                    )
                    or (
                        self.get_head_position().x == self.body[i].x
                        and self.get_head_position().y - self.body[i].y == -1
                        and self.y_direction == 1
                    )
                    or (
                        self.get_head_position().y == self.body[i].y
                        and self.get_head_position().x - self.body[i].x == 1
                        and self.x_direction == -1
                    )
                    or (
                        self.get_head_position().y == self.body[i].y
                        and self.get_head_position().x - self.body[i].x == -1
                        and self.x_direction == 1
                    )
                ):
                    print("Collision with body")
                new_position = apple_instance.generate(self.body)
                if new_position is None:
                    self.show()  # This might be a method to display some state when no new apple can be placed.
                else:
                    apple_instance.position = new_position
                    self.body.pop(
                        0
                    )  # Continue with the game logic if a new position is generated.
                self.body.append(
                    Vector2(
                        self.get_head_position().x + self.x_direction,
                        self.get_head_position().y + self.y_direction,
                    )
                )

    def get_head_position(self) -> Vector2:
        return self.body[-1]

    def get_tail_position(self) -> Vector2:
        return self.body[0]

    def change_direction(self, x_direction: int, y_direction: int) -> None:
        if not (
            abs(self.x_direction - x_direction) == 2
            or abs(self.y_direction - y_direction) == 2
        ):
            self.x_direction = x_direction
            self.y_direction = y_direction

    def move_up(self) -> None:
        self.change_direction(0, -1)

    def move_down(self) -> None:
        self.change_direction(0, 1)

    def move_left(self) -> None:
        self.change_direction(-1, 0)

    def move_right(self) -> None:
        self.change_direction(1, 0)

    def show(self, screen: pg.Surface) -> None:
        for i in range(len(self.body)):
            pg.draw.rect(
                screen,
                (0, 164, 239),
                (self.body[i].x * 30, self.body[i].y * 30, 30, 30),
            )
            if i == 0:
                back_toggle = 1
            else:
                back_toggle = -1
            if i == len(self.body) - 1:
                front_toggle = -1
            else:
                front_toggle = 1
            if not (
                self.body[i].x == self.body[i + back_toggle].x
                and self.body[i].y - self.body[i + back_toggle].y == 1
            ):
                if not (
                    self.body[i].x == self.body[i + front_toggle].x
                    and self.body[i].y - self.body[i + front_toggle].y == 1
                ):
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30, self.body[i].y * 30),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30),
                    )
            if not (
                self.body[i].x == self.body[i + back_toggle].x
                and self.body[i].y - self.body[i + back_toggle].y == -1
            ):
                if not (
                    self.body[i].x == self.body[i + front_toggle].x
                    and self.body[i].y - self.body[i + front_toggle].y == -1
                ):
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30, self.body[i].y * 30 + 30),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30 + 30),
                    )
            if not (
                self.body[i].y == self.body[i + back_toggle].y
                and self.body[i].x - self.body[i + back_toggle].x == -1
            ):
                if not (
                    self.body[i].y == self.body[i + front_toggle].y
                    and self.body[i].x - self.body[i + front_toggle].x == -1
                ):
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30 + 30),
                    )
            if not (
                self.body[i].y == self.body[i + back_toggle].y
                and self.body[i].x - self.body[i + back_toggle].x == 1
            ):
                if not (
                    self.body[i].y == self.body[i + front_toggle].y
                    and self.body[i].x - self.body[i + front_toggle].x == 1
                ):
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30, self.body[i].y * 30),
                        (self.body[i].x * 30, self.body[i].y * 30 + 30),
                    )


def main() -> None:
    pg.init()
    screen: pg.Surface = pg.display.set_mode(screen_size)
    apple_object: apple.Apple = apple_instance
    snake: Snake = Snake()
    search_algorithm: SearchAlgorithm = SearchAlgorithm(snake, apple_object)
    search_algorithm.get_path()
    clock: pg.time.Clock = pg.time.Clock()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        snake.update(apple_object)
        snake.show(screen)
        apple_object.show(screen)
        pg.display.flip()
        clock.tick(frames_per_second)


if __name__ == "__main__":
    main()
