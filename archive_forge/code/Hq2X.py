import pygame
import sys
import logging
from typing import Optional, List, Tuple

from Constants import *
from Menu import MainMenu, GAMenu
from GameController import GameController
from GA import GA, Population

# Setting up logging configuration with detailed debug information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GameGUI:
    """
    This class represents the graphical user interface for the snake game.
    It handles all the visual elements, game loop, and interactions with the user.
    """

    def __init__(self):
        """
        Initializes the GameGUI object, setting up the pygame environment,
        logging, and game components like menus and controllers.
        """
        # Initialize the pygame module and log the initialization status
        pygame.init()
        logging.debug("Pygame initialized with all modules loaded.")

        # Clock to manage game frame rate and a custom event for screen updates
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.SCREEN_UPDATE: int = pygame.USEREVENT
        logging.debug(
            f"Clock and screen update event initialized with ID: {self.SCREEN_UPDATE}"
        )

        # Speed settings for the game loop and speed adjustment
        self.speed: int = 110
        self.speed_up: int = 80
        logging.debug(
            f"Initial speed set to {self.speed} and speed up increment set to {self.speed_up}"
        )

        pygame.time.set_timer(self.SCREEN_UPDATE, self.speed)
        logging.debug(f"Screen update timer set with speed {self.speed}")

        self.controller: GameController = GameController()
        logging.debug("GameController instance created.")

        self.running: bool = True
        self.playing: bool = False
        self.UPKEY: bool = False
        self.DOWNKEY: bool = False
        self.START: bool = False
        self.BACK: bool = False
        logging.debug("Initial state flags set.")

        self.SIZE: int = CELL_SIZE * NO_OF_CELLS
        self.display: pygame.Surface = pygame.Surface((self.SIZE, self.SIZE))
        self.window: pygame.display = pygame.display.set_mode((self.SIZE, self.SIZE))
        logging.debug(f"Display and window initialized with size {self.SIZE}.")

        self.font_name: str = None
        self.font: pygame.font.Font = pygame.font.SysFont(self.font_name, 30)
        logging.debug(f"Font set to {self.font_name}")

        self.main_menu: MainMenu = MainMenu(self)
        self.GA: GAMenu = GAMenu(self, self.controller)
        self.curr_menu = self.main_menu
        logging.debug("Menus initialized and current menu set to main menu.")

        self.load_model: bool = False
        self.view_path: bool = False
        logging.debug("Model loading and path viewing flags initialized.")

    def game_loop(self):
        """
        The main game loop that runs while the game is in the 'playing' state.
        Handles events, updates the display, and manages game state transitions.
        """
        while self.playing:
            logging.debug("Game loop started.")
            self.event_handler()

            if self.BACK:
                self.playing = False
                logging.debug("Back key pressed, stopping game loop.")

            self.display.fill(WINDOW_COLOR)
            logging.debug("Display filled with window color.")

            if self.controller.algo is not None:
                self.draw_elements()
                logging.debug("Game elements drawn.")

            self.window.blit(self.display, (0, 0))
            pygame.display.update()
            self.clock.tick(60)
            logging.debug("Display updated and clock ticked.")

            self.reset_keys()
            logging.debug("Keys reset.")

    def draw_elements(self):
        """
        Draws all game elements on the screen, including the banner, game stats,
        and depending on the game state, either the snake and fruit or all GA snakes.
        """
        # draw banner and stats
        self.draw_banner()
        self.draw_game_stats()
        logging.debug("Banner and game stats drawn.")

        if (
            self.curr_menu.state != "GA" or self.controller.model_loaded
        ):  # Path Ai or trained GA
            fruit = self.controller.get_fruit_pos()
            snake = self.controller.snake

            self.draw_fruit(fruit)
            self.draw_snake(snake)
            self.draw_score()
            logging.debug("Fruit, snake, and score drawn.")

            if not self.controller.model_loaded:
                self.draw_path()  # only path Ai has a path
                logging.debug("Path drawn.")

        else:  # training a GA model
            self.draw_all_snakes_GA()
            logging.debug("All GA snakes drawn.")

    def draw_game_stats(self):
        """
        Draws the game statistics and instructions on the screen based on the current game state.
        """
        if self.curr_menu.state != "GA":  # path Ai algo
            instruction: str = "Space to view Ai path, W to speed up, Q to go back"
            logging.debug("Instructions for path AI set.")

        elif self.controller.model_loaded:  # trained model
            instruction: str = "W to speed up, Q to go back"
            logging.debug("Instructions for trained model set.")

        else:  # training model GA algo
            instruction: str = "Space to hide all snakes, W to speed up, Q to go back"
            curr_gen: str = str(self.controller.curr_gen())
            best_score: str = str(self.controller.best_GA_score())

            stats_gen: str = f"Generation: {curr_gen}/{GA.generation}"
            stats_score: str = f"Best score: {best_score}"
            stats_hidden_node: str = f"Hidden nodes {Population.hidden_node}"
            logging.debug(
                f"GA training instructions and stats set: {stats_gen}, {stats_score}, {stats_hidden_node}"
            )

            # draw stats
            self.draw_text(
                stats_gen,
                size=20,
                x=3 * CELL_SIZE,
                y=CELL_SIZE - 10,
            )
            self.draw_text(
                stats_score,
                size=20,
                x=3 * CELL_SIZE,
                y=CELL_SIZE + 20,
            )
            self.draw_text(
                stats_hidden_node,
                size=20,
                x=self.SIZE / 2,
                y=CELL_SIZE - 30,
                color=SNAKE_COLOR,
            )
            logging.debug("GA training stats drawn.")

        # instruction
        self.draw_text(
            instruction,
            size=20,
            x=self.SIZE / 2,
            y=(CELL_SIZE * NO_OF_CELLS) - NO_OF_CELLS,
            color=WHITE,
        )
        logging.debug("Instructions drawn.")

        # current Algo Title
        self.draw_text(
            self.curr_menu.state,
            size=30,
            x=self.SIZE / 2,
            y=CELL_SIZE,
        )
        logging.debug("Current algorithm title drawn.")

    def draw_all_snakes_GA(self):
        """
        Draws all snakes during the GA training phase, unless the view path flag is toggled.
        """
        if not self.view_path:  # have all snakes visible by default

            for snake in self.controller.snakes:  # for each snake in list
                self.draw_snake(snake)

                # fruit of each snake
                self.draw_fruit(snake.get_fruit())
                logging.debug(f"Snake and its fruit drawn for GA training: {snake}")

    def draw_path(self):
        """
        Draws the path of the AI snake if the view path flag is toggled and an algorithm is active.
        """
        if self.controller.algo is not None and self.view_path:
            for path in self.controller.algo.path:  # for each {x,y} in path
                x: int = int(path.x * CELL_SIZE)
                y: int = int(path.y * CELL_SIZE)

                path_rect: pygame.Rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                shape_surf: pygame.Surface = pygame.Surface(
                    path_rect.size, pygame.SRCALPHA
                )
                pygame.draw.rect(shape_surf, PATHCOLOR, shape_surf.get_rect())

                pygame.draw.rect(self.display, BANNER_COLOR, path_rect, 1)
                self.display.blit(shape_surf, path_rect)
                logging.debug(f"Path segment drawn at {x}, {y}.")

    def draw_snake_head(self, snake):
        """
        Draws the head of the snake on the display.
        """
        head = snake.body[0]
        self.draw_rect(head, color=SNAKE_HEAD_COLOR)
        logging.debug(f"Snake head drawn at {head}.")

    def draw_snake_body(self, body):
        """
        Draws a body segment of the snake on the display.
        """
        self.draw_rect(body, color=SNAKE_COLOR, border=True)
        logging.debug(f"Snake body segment drawn at {body}.")

    def draw_rect(self, element, color, border=False):
        """
        Draws a rectangle on the display representing either a snake body segment or the snake head.
        """
        x: int = int(element.x * CELL_SIZE)
        y: int = int(element.y * CELL_SIZE)

        body_rect: pygame.Rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.display, color, body_rect)

        if border:
            pygame.draw.rect(self.display, WINDOW_COLOR, body_rect, 3)
            logging.debug(f"Rect drawn with border at {x}, {y}.")

    def draw_snake(self, snake):
        """
        Draws the entire snake on the display, including its head and body segments.
        """
        self.draw_snake_head(snake)  # draw head
        logging.debug("Snake head drawn.")

        for body in snake.body[1:]:
            self.draw_snake_body(body)  # draw body
            logging.debug("Snake body drawn.")

    def draw_fruit(self, fruit):
        """
        Draws the fruit on the display, which the snake aims to eat.
        """
        x: int = int(fruit.x * CELL_SIZE)
        y: int = int(fruit.y * CELL_SIZE)

        fruit_rect: pygame.Rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.display, FRUIT_COLOR, fruit_rect)
        logging.debug(f"Fruit drawn at {x}, {y}.")

    def draw_banner(self):
        """
        Draws the banner at the top of the display.
        """
        banner: pygame.Rect = pygame.Rect(0, 0, self.SIZE, BANNER_HEIGHT * CELL_SIZE)
        pygame.draw.rect(self.display, BANNER_COLOR, banner)
        logging.debug("Banner drawn.")

    def draw_score(self):
        """
        Draws the current score of the game on the display.
        """
        score_text: str = "Score: " + str(self.controller.get_score())
        score_x: int = self.SIZE - (CELL_SIZE + 2 * len(score_text))
        score_y: int = CELL_SIZE
        self.draw_text(score_text, 20, score_x, score_y, WINDOW_COLOR)
        logging.debug("Score drawn.")

    def game_over(self):
        """
        Handles the game over state, allowing the user to save the model or restart the game.
        """
        again: bool = False

        while not again:
            for event in pygame.event.get():
                if self.is_quit(event):
                    again = True
                    pygame.quit()
                    sys.exit()
                    logging.debug("Game quit.")

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        again = True
                        break
                    if event.key == pygame.K_s:
                        again = True
                        self.controller.save_model()
                        break
                    logging.debug(f"Key pressed: {pygame.key.name(event.key)}")

            self.display.fill(MENU_COLOR)

            # training model results
            if self.curr_menu.state == "GA" and not self.controller.model_loaded:
                best_score: int = self.controller.best_GA_score()
                best_gen: int = self.controller.best_GA_gen()

                high_score: str = (
                    f"Best snake Score: {best_score} in generation {best_gen}"
                )
                save: str = "Press S to save best snake"

                self.draw_text(
                    save,
                    size=30,
                    x=self.SIZE / 2,
                    y=self.SIZE / 2 + 3 * CELL_SIZE,
                    color=FRUIT_COLOR,
                )
            else:
                # Path ai or trained model results
                high_score: str = f"High Score: {self.controller.get_score()}"

            to_continue: str = "Enter to Continue"

            self.draw_text(
                high_score,
                size=35,
                x=self.SIZE / 2,
                y=self.SIZE / 2,
            )

            self.draw_text(
                to_continue,
                size=30,
                x=self.SIZE / 2,
                y=self.SIZE / 2 + 2 * CELL_SIZE,
                color=WHITE,
            )

            self.window.blit(self.display, (0, 0))
            pygame.display.update()
            logging.debug("Game over screen updated.")
        self.controller.reset()
        logging.debug("Game controller reset.")

    def is_quit(self, event):
        """
        Checks if the quit event is triggered by the user.
        """
        # user presses exit icon
        if event.type == pygame.QUIT:
            self.running, self.playing = False, False
            self.curr_menu.run_display = False
            logging.debug("Quit event detected.")
            return True
        return False

    def event_handler(self):
        """
        Handles all events during the game loop, including key presses and custom events.
        """
        for event in pygame.event.get():
            if self.is_quit(event):
                print("Bye :)")
                pygame.quit()
                sys.exit()
                logging.debug("Event handler quit.")

            # user event that runs every self.speed milisec
            elif self.playing and event.type == pygame.USEREVENT:

                if self.load_model:  # user load model
                    self.controller.load_model()
                    self.load_model = False
                    logging.debug("Model loaded.")

                self.controller.ai_play(self.curr_menu.state)  # play

                if self.controller.end == True:  # Only path ai and trained model
                    self.playing = False
                    self.game_over()  # show game over stats
                    logging.debug("Game over initiated.")

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RETURN:  # on Enter
                    self.START = True
                    self.view_path = False
                    logging.debug("Enter key pressed.")

                elif event.key == pygame.K_q:  # on q return
                    self.BACK = True
                    self.controller.reset()
                    logging.debug("Q key pressed and controller reset.")

                elif (
                    event.key == pygame.K_SPACE
                ):  # space view path or hide training snakes
                    self.view_path = not self.view_path
                    logging.debug("Space key toggled view path.")

                elif event.key == pygame.K_DOWN:
                    self.DOWNKEY = True
                    logging.debug("Down key pressed.")

                elif event.key == pygame.K_UP:
                    self.UPKEY = True
                    logging.debug("Up key pressed.")

                elif event.key == pygame.K_w:  # speed up/down by self.speed_up
                    self.speed_up = -1 * self.speed_up
                    self.speed = self.speed + self.speed_up
                    pygame.time.set_timer(self.SCREEN_UPDATE, self.speed)
                    logging.debug(f"Speed adjusted to {self.speed}.")

    def reset_keys(self):
        """
        Resets all key flags to False after handling them in the game loop.
        """
        self.UPKEY, self.DOWNKEY, self.START, self.BACK = False, False, False, False
        logging.debug("All keys reset.")

    def draw_text(self, text, size, x, y, color=WINDOW_COLOR):
        """
        Draws text on the display at the specified location and with the specified properties.
        """
        font = pygame.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        self.display.blit(text_surface, text_rect)
        logging.debug(f"Text '{text}' drawn at {x}, {y}.")
