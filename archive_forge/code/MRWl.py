# // Creating the snake, apple and A* search algorithm
# Creating the snake, apple and search algorithm

# let apple;
import apple

from search import Search

# Rest of the Imports required in alignment with the rest of the classes.
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


# Calculate the block size based on screen resolution to ensure visibility and proportionality
# Define a scaling function for block size relative to screen resolution
def calculate_block_size(screen_width: int, screen_height: int) -> int:
    # Define the reference resolution and corresponding block size
    reference_resolution = (1920, 1080)
    reference_block_size = 20

    # Calculate the scaling factor based on the reference
    scaling_factor_width = screen_width / reference_resolution[0]
    scaling_factor_height = screen_height / reference_resolution[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    # Calculate the block size dynamically based on the screen size
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))

    # Ensure the block size does not become too large or too small
    # Set minimum block size to 1x1 pixels and maximum to 30x30 pixels
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


# Apply the calculated block size based on the current screen resolution
BLOCK_SIZE = calculate_block_size(display_info.current_w, display_info.current_h)

# Define the border width as equivalent to 3 blocks
border_width = 3 * BLOCK_SIZE  # Width of the border to be subtracted from each side

# Define the screen size with a proportional border around the edges
SCREEN_SIZE = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)

# Define a constant for the border color as solid white
BORDER_COLOR = (255, 255, 255)  # RGB color code for white
APPLE = apple.Apple()
CLOCK = pg.time.Clock()
FPS = 60
TICK_RATE = 1000 // FPS


# // Setting up everything
# Initial setup of the game environment
# function setup() {
def setup() -> Tuple[pg.Surface, apple.Apple, Search, pg.time.Clock]:
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    # Initialize Pygame
    pg.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pg.Surface = pg.display.set_mode(SCREEN_SIZE)
    # Instantiate the Apple object using the globally defined APPLE
    apple: apple.Apple = APPLE
    # Instantiate the Search algorithm object
    search: Search = Search()
    # instantiatie the path
    search.get_path()
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock: pg.time.Clock = CLOCK
    return screen, apple, search, clock


class Snake:
    # // Creating the snake object
    # constructor() {
    def __init__(self):  # Python class constructor
        # this.body = [];
        self.body: List[Vector2] = []  # Python list initialization with type annotation
        # // The snake starts with a length of 3 from the top-left corner
        # // Max length for the snake is 800 pixels --> perfection!
        # for (let i = 0; i < 3; i++) {
        #     this.body[i] = createVector(i, 0);
        # }
        for i in range(3):  # Python for loop
            self.body.append(Vector2(i, 0))  # Using pygame Vector2 for position
        # this.x_dir = 1;
        self.x_dir: int = 1  # Python integer type annotation
        # this.y_dir = 0;
        self.y_dir: int = 0  # Python integer type annotation
        # this.path = [];
        self.path: List[Vector2] = []  # Python list initialization with type annotation
        # this.tail_position = this.getTailPosition();
        self.tail_position: Vector2 = (
            self.get_tail_position()
        )  # Method call in Python with type annotation

    # // Updating the snake position
    # update(apple) {
    def update(
        self, apple: apple.Apple
    ) -> None:  # Python method definition with type annotations
        # // The path is refreshed using A* if the head reached the previous tail position
        # // or when the snake's length reached 600 pixels out of 800 pixels
        # if ((this.getHeadPosition().x == this.tail_position.x && this.getHeadPosition().y == this.tail_position.y) || this.body.length > 600) {
        if (
            self.get_head_position().x == self.tail_position.x
            and self.get_head_position().y == self.tail_position.y
        ) or len(
            self.body
        ) > 600:  # Python logical and comparison operators
            # this.tail_position = createVector(0, 0);
            self.tail_position = Vector2(0, 0)  # Using pygame Vector2 for position
            # this.tail_position.x = this.getTailPosition().x;
            self.tail_position.x = (
                self.get_tail_position().x
            )  # Accessing Vector2 x attribute
            # this.tail_position.y = this.getTailPosition().y;
            self.tail_position.y = (
                self.get_tail_position().y
            )  # Accessing Vector2 y attribute
            # search.getPath();
            Search.get_path()  # Method call in Python

        # // The direction of the snake set using the path generated from A*
        # for (let i = 0; i < this.path.length; i++) {
        for i in range(len(self.path)):  # Python for loop
            # let head = this.getHeadPosition();
            head: Vector2 = (
                self.get_head_position()
            )  # Local variable assignment with type annotation
            # if (head.x == this.path[i].x && head.y == this.path[i].y) {
            if (
                head.x == self.path[i].x and head.y == self.path[i].y
            ):  # Python logical and comparison operators
                # let next_head = this.path[i + 1];
                next_head: Vector2 = self.path[
                    i + 1
                ]  # Accessing list element by index with type annotation
                # if (next_head.x - head.x == 1) {
                if next_head.x - head.x == 1:  # Comparison operator
                    # this.right();
                    self.right()  # Method call in Python
                # else if (next_head.x - head.x == -1) {
                elif next_head.x - head.x == -1:  # Python elif statement
                    # this.left();
                    self.left()  # Method call in Python
                # else if (next_head.y - head.y == 1) {
                elif next_head.y - head.y == 1:  # Comparison operator
                    # this.down();
                    self.down()  # Method call in Python
                # else if (next_head.y - head.y == -1) {
                elif next_head.y - head.y == -1:  # Comparison operator
                    # this.up();
                    self.up()  # Method call in Python
                # else {
                else:  # Python else statement
                    # console.log("Something is wrong");
                    print("Something is wrong")  # Python print function
                    # noLoop();

        # // Collision with wall logic
        # if (this.getHeadPosition().x == 39 && this.x_dir == 1) {
        if (
            self.get_head_position().x == 39 and self.x_dir == 1
        ):  # Python logical and comparison operators
            # noLoop();
            # console.log("Collision with wall");
            print("Collision with wall")  # Python print function
        # else if (this.getHeadPosition().x == 0 && this.x_dir == -1) {
        elif (
            self.get_head_position().x == 0 and self.x_dir == -1
        ):  # Python elif statement
            # noLoop();
            # console.log("Collision with wall");
            print("Collision with wall")  # Python print function
        # else if (this.getHeadPosition().y == 19 && this.y_dir == 1) {
        elif (
            self.get_head_position().y == 19 and self.y_dir == 1
        ):  # Python elif statement
            # noLoop();
            # console.log("Collision with wall");
            print("Collision with wall")  # Python print function
        # else if (this.getHeadPosition().y == 0 && this.y_dir == -1) {
        elif (
            self.get_head_position().y == 0 and self.y_dir == -1
        ):  # Python elif statement
            # noLoop();
            # console.log("Collision with wall");
            print("Collision with wall")  # Python print function
        # else {
        else:  # Python else statement
            # // Collision logic with the snake body
            # for (let i = 0; i < this.body.length - 1; i += 1) {
            for i in range(len(self.body) - 1):  # Python for loop
                # if (this.getHeadPosition().x == this.body[i].x && this.getHeadPosition().y - this.body[i].y == 1 && this.y_dir == -1) {
                if (
                    self.get_head_position().x == self.body[i].x
                    and self.get_head_position().y - self.body[i].y == 1
                    and self.y_dir == -1
                ):  # Python logical and comparison operators
                    # noLoop();
                    # console.log("Collision with body");
                    print("Collision with body")  # Python print function
                # else if (this.getHeadPosition().x == this.body[i].x && this.getHeadPosition().y - this.body[i].y == -1 && this.y_dir == 1) {
                elif (
                    self.get_head_position().x == self.body[i].x
                    and self.get_head_position().y - self.body[i].y == -1
                    and self.y_dir == 1
                ):  # Python elif statement
                    # noLoop();
                    # console.log("Collision with body");
                    print("Collision with body")  # Python print function
                # else if (this.getHeadPosition().y == this.body[i].y && this.getHeadPosition().x - this.body[i].x == 1 && this.x_dir == -1) {
                elif (
                    self.get_head_position().y == self.body[i].y
                    and self.get_head_position().x - self.body[i].x == 1
                    and self.x_dir == -1
                ):  # Python elif statement
                    # noLoop();
                    # console.log("Collision with body");
                    print("Collision with body")  # Python print function
                # else if (this.getHeadPosition().y == this.body[i].y && this.getHeadPosition().x - this.body[i].x == -1 && this.x_dir == 1) {
                elif (
                    self.get_head_position().y == self.body[i].y
                    and self.get_head_position().x - self.body[i].x == -1
                    and self.x_dir == 1
                ):  # Python elif statement
                    # noLoop();
                    # console.log("Collision with body");
                    print("Collision with body")  # Python print function

                # // The snake body is elongated at its head using the direction
                # this.body.push(createVector(this.getHeadPosition().x + this.x_dir, this.getHeadPosition().y + this.y_dir));
                self.body.append(
                    Vector2(
                        self.get_head_position().x + self.x_dir,
                        self.get_head_position().y + self.y_dir,
                    )
                )  # Adding a new position to the body

                # // A new apple position is generated when the snake is about
                # // to eat the apple
                # if (
                #     (this.getHeadPosition().x == apple.x && this.getHeadPosition().y - apple.y == 0 && this.y_dir == -1) ||
                #     (this.getHeadPosition().x == apple.x && this.getHeadPosition().y - apple.y == 0 && this.y_dir == 1) ||
                #     (this.getHeadPosition().y == apple.y && this.getHeadPosition().x - apple.x == 0 && this.x_dir == -1) ||
                #     (this.getHeadPosition().y == apple.y && this.getHeadPosition().x - apple.x == 0 && this.x_dir == 1)
                # ) {
                if (
                    (
                        self.get_head_position().x == APPLE.x
                        and self.get_head_position().y - APPLE.y == 0
                        and self.y_dir == -1
                    )
                    or (
                        self.get_head_position().x == APPLE.x
                        and self.get_head_position().y - APPLE.y == 0
                        and self.y_dir == 1
                    )
                    or (
                        self.get_head_position().y == APPLE.y
                        and self.get_head_position().x - APPLE.x == 0
                        and self.x_dir == -1
                    )
                    or (
                        self.get_head_position().y == APPLE.y
                        and self.get_head_position().x - APPLE.x == 0
                        and self.x_dir == 1
                    )
                ):  # Python logical and comparison operators
                    # if (!apple.generate(this.body)){
                    if not APPLE.generate(self.body):  # Python not operator
                        # snake.show();
                        self.show()  # Method call in Python
                        # noLoop();
                # }
                # // If apple is not eaten then the snake tail is cut
                # // to keep the same snake length
                # else {
                else:  # Python else statement
                    # this.body.splice(0, 1);
                    self.body.pop(0)  # Removing the first element from the list

        # }

    # getHeadPosition() {
    def get_head_position(
        self,
    ) -> Vector2:  # Python method definition with return type annotation
        # return this.body[this.body.length - 1];
        return self.body[-1]  # Python negative indexing

    # getTailPosition() {
    def get_tail_position(
        self,
    ) -> Vector2:  # Python method definition with return type annotation
        # return this.body[0];
        return self.body[0]  # Python list indexing

    # // Direction is changed
    def change_direction(self, x_dir: int, y_dir: int) -> None:
        """
        Change the direction of the snake.

        Args:
            x_dir (int): The new x-direction of the snake.
            y_dir (int): The new y-direction of the snake.

        Returns:
            None
        """
        # if (!(abs(this.x_dir - x_dir) == 2 || abs(this.y_dir - y_dir) == 2)) {
        if not (abs(self.x_dir - x_dir) == 2 or abs(self.y_dir - y_dir) == 2):
            # this.x_dir = x_dir;
            self.x_dir = x_dir
            # this.y_dir = y_dir;
            self.y_dir = y_dir

    # // Direction functions
    def up(self) -> None:
        """
        Change the direction of the snake to move upwards.

        Returns:
            None
        """
        # this.changeDirection(0, -1);
        self.change_direction(0, -1)

    def down(self) -> None:
        """
        Change the direction of the snake to move downwards.

        Returns:
            None
        """
        # this.changeDirection(0, 1);
        self.change_direction(0, 1)

    def left(self) -> None:
        """
        Change the direction of the snake to move left.

        Returns:
            None
        """
        # this.changeDirection(-1, 0);
        self.change_direction(-1, 0)

    def right(self) -> None:
        """
        Change the direction of the snake to move right.

        Returns:
            None
        """
        # this.changeDirection(1, 0);
        self.change_direction(1, 0)

    def show(self, screen: pg.Surface) -> None:
        """
        Display the snake on the screen.

        Returns:
            None
        """
        # fill(0, 164, 239);
        pg.draw.rect(
            screen, (0, 164, 239), (self.body[0].x * 30, self.body[0].y * 30, 30, 30)
        )
        # strokeWeight(0);
        # rect(this.body[0].x * 30, this.body[0].y * 30, 30, 30);
        pg.draw.rect(
            screen, (0, 164, 239), (self.body[0].x * 30, self.body[0].y * 30, 30, 30)
        )
        # stroke(51);
        # strokeWeight(2);
        # let backToggle = -1;
        back_toggle: int = -1
        # let frontToggle = 1;
        front_toggle: int = 1
        # // This looks quite overwhelming but basically lines are drawn where
        # // boxes do not have a common border
        # // This draws an outline around the snake which has the same colour as the background
        # // This makes the snake look nice
        # for (let i = 0; i < this.body.length; i++) {
        for i in range(len(self.body)):
            # strokeWeight(0);
            # rect(this.body[i].x * 30, this.body[i].y * 30, 30, 30);
            pg.draw.rect(
                screen,
                (0, 164, 239),
                (self.body[i].x * 30, self.body[i].y * 30, 30, 30),
            )
            # stroke(51);
            # strokeWeight(2);
            # if (i == 0) {
            if i == 0:
                # backToggle = 1;
                back_toggle = 1
            # } else {
            else:
                # backToggle = -1;
                back_toggle = -1
            # }
            # if (i == this.body.length - 1) {
            if i == len(self.body) - 1:
                # frontToggle = -1;
                front_toggle = -1
            # } else {
            else:
                # frontToggle = 1;
                front_toggle = 1
            # }
            # if (!(this.body[i].x == this.body[i + backToggle].x && this.body[i].y - this.body[i + backToggle].y == 1)) {
            if not (
                self.body[i].x == self.body[i + back_toggle].x
                and self.body[i].y - self.body[i + back_toggle].y == 1
            ):
                # if (!(this.body[i].x == this.body[i + frontToggle].x && this.body[i].y - this.body[i + frontToggle].y == 1)) {
                if not (
                    self.body[i].x == self.body[i + front_toggle].x
                    and self.body[i].y - self.body[i + front_toggle].y == 1
                ):
                    # line(this.body[i].x * 30, this.body[i].y * 30, this.body[i].x * 30 + 30, this.body[i].y * 30);
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30, self.body[i].y * 30),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30),
                    )
            # }
            # if (!(this.body[i].x == this.body[i + backToggle].x && this.body[i].y - this.body[i + backToggle].y == -1)) {
            if not (
                self.body[i].x == self.body[i + back_toggle].x
                and self.body[i].y - self.body[i + back_toggle].y == -1
            ):
                # if (!(this.body[i].x == this.body[i + frontToggle].x && this.body[i].y - this.body[i + frontToggle].y == -1)) {
                if not (
                    self.body[i].x == self.body[i + front_toggle].x
                    and self.body[i].y - self.body[i + front_toggle].y == -1
                ):
                    # line(this.body[i].x * 30, this.body[i].y * 30 + 30, this.body[i].x * 30 + 30, this.body[i].y * 30 + 30);
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30, self.body[i].y * 30 + 30),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30 + 30),
                    )
            # }
            # if (!(this.body[i].y == this.body[i + backToggle].y && this.body[i].x - this.body[i + backToggle].x == -1)) {
            if not (
                self.body[i].y == self.body[i + back_toggle].y
                and self.body[i].x - self.body[i + back_toggle].x == -1
            ):
                # if (!(this.body[i].y == this.body[i + frontToggle].y && this.body[i].x - this.body[i + frontToggle].x == -1)) {
                if not (
                    self.body[i].y == self.body[i + front_toggle].y
                    and self.body[i].x - self.body[i + front_toggle].x == -1
                ):
                    # line(this.body[i].x * 30 + 30, this.body[i].y * 30, this.body[i].x * 30 + 30, this.body[i].y * 30 + 30);
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30),
                        (self.body[i].x * 30 + 30, self.body[i].y * 30 + 30),
                    )
            # }
            # if (!(this.body[i].y == this.body[i + backToggle].y && this.body[i].x - this.body[i + backToggle].x == 1)) {
            if not (
                self.body[i].y == self.body[i + back_toggle].y
                and self.body[i].x - self.body[i + back_toggle].x == 1
            ):
                # if (!(this.body[i].y == this.body[i + frontToggle].y && this.body[i].x - this.body[i + frontToggle].x == 1)) {
                if not (
                    self.body[i].y == self.body[i + front_toggle].y
                    and self.body[i].x - self.body[i + front_toggle].x == 1
                ):
                    # line(this.body[i].x * 30, this.body[i].y * 30, this.body[i].x * 30, this.body[i].y * 30 + 30);
                    pg.draw.line(
                        screen,
                        (51, 51, 51),
                        (self.body[i].x * 30, self.body[i].y * 30),
                        (self.body[i].x * 30, self.body[i].y * 30 + 30),
                    )
            # }
        # }

    # }


def main() -> None:
    """
    This function is run when the main script is run as the main program.
    It acts as a comprehensive, programmatic, simulated set of unit tests to
    ensure that everything in the snake.py file is working as expected.
    """
    # Initialize Pygame
    pg.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pg.Surface = pg.display.set_mode(SCREEN_SIZE)
    # Instantiate the Apple object using the globally defined APPLE
    apple: apple.Apple = APPLE
    # Instantiate the Snake object
    SNAKE: Snake = Snake()
    # Instantiate the Search algorithm object
    search: Search = Search(SNAKE, APPLE)
    # instantiatie the path
    search.get_path()
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock: pg.time.Clock = CLOCK

    # Run the game loop
    while True:
        # Check for Pygame events
        for event in pg.event.get():
            # Check if the event is a quit event
            if event.type == pg.QUIT:
                # Quit Pygame
                pg.quit()

        # Update the snake
        SNAKE.update(apple)
        # Draw the snake on the screen
        SNAKE.show(screen)
        # Display the apple on the screen
        apple.show(screen)
        # Update the display
        pg.display.flip()
        # Tick the clock to control the frame rate
        clock.tick(FPS)


if __name__ == "__main__":
    main()
