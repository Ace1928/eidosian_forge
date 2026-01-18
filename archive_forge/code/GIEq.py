# // Creating the snake, apple and A* search algorithm
# Creating the snake, apple and search algorithm
# let snake;
import snake

# let apple;
import apple

# let search;
import search

# Rest of the Imports required in alignment with the rest of the classes.
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint


# Retrieve the current display information
display_info = pg.display.Info()

# Define the screen size with a small border around the edges
border_width = 50  # Width of the border to be subtracted from each side
SCREEN_SIZE = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)

SNAKE = snake.Snake()
APPLE = apple.Apple()


# // Setting up everything
# Initial setup of the game environment
# function setup() {
def setup():
    #    createCanvas(1200, 600);
    pg.init()
    screen = pg.display.set_mode((1200, 600))
    #    snake = new Snake();
    snake = snake.Snake()
    #    apple = new Apple();
    apple = apple.Apple()
    #    search = new Search(snake, apple);
    search = search.Search(snake, apple)
    #    search.getPath();
    search.get_path()
    #    frameRate(200);
    clock = pg.time.Clock()
    return screen, snake, apple, search, clock


# }
# This functiona manages the drawing of all game objects on the screen.
# function draw() {
def draw(screen, snake, apple, clock):
    #    background(51);
    screen.fill((51, 51, 51))
    #    snake.show();
    snake.show()
    #    apple.show();
    apple.draw(screen)
    #    snake.update(apple);
    snake.update(apple)
    # }
    pg.display.flip()
    clock.tick(200)
