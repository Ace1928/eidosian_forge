import pygame
import sys
import logging
from typing import Tuple, Union
from Constants import *
from GA import *

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Menu:
    """
    A class to represent the menu interface of the game.

    Attributes:
        game (Game): The game instance.
        mid_size (float): Half the size of the game window.
        run_display (bool): A flag to determine if the menu should continue displaying.
        cursor_rect (pygame.Rect): The rectangle representing the cursor.
        offset (int): The offset for cursor positioning.
        title_size (int): Font size for the title.
        option_size (int): Font size for menu options.
    """

    def __init__(self, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the menu object.

        Parameters:
            game (Game): The game instance.
        """
        logging.debug("Initializing Menu class with game instance.")
        self.game: "Game" = game
        self.mid_size: float = self.game.SIZE / 2
        self.run_display: bool = True
        self.cursor_rect: pygame.Rect = pygame.Rect(0, 0, 20, 20)
        self.offset: int = -150
        self.title_size: int = 50
        self.option_size: int = 28
        logging.debug(
            f"Menu class initialized with mid_size: {self.mid_size}, run_display: {self.run_display}, cursor_rect: {self.cursor_rect}, offset: {self.offset}, title_size: {self.title_size}, option_size: {self.option_size}."
        )

    def draw_cursor(self) -> None:
        """
        Draws the cursor on the menu.
        """
        logging.debug("Attempting to draw cursor on menu.")
        try:
            self.game.draw_text(
                "*",
                size=20,
                x=self.cursor_rect.x,
                y=self.cursor_rect.y,
                color=MENU_COLOR,
            )
            logging.debug(
                f"Cursor successfully drawn at position: ({self.cursor_rect.x}, {self.cursor_rect.y})."
            )
        except Exception as e:
            logging.error(f"Failed to draw cursor due to: {e}", exc_info=True)
            raise RuntimeError(f"Drawing cursor failed due to: {e}") from e

    def blit_menu(self) -> None:
        """
        Blits the menu to the screen.
        """
        logging.debug("Attempting to blit menu to the screen.")
        try:
            self.game.window.blit(self.game.display, (0, 0))
            pygame.display.update()
            self.game.reset_keys()
            logging.debug("Menu blitted successfully to the screen.")
        except Exception as e:
            logging.error(f"Failed to blit menu due to: {e}", exc_info=True)
            raise RuntimeError(f"Blitting menu failed due to: {e}") from e


class MainMenu(Menu):
    """
    A class to represent the main menu of the game, inheriting from Menu.

    Attributes:
        state (str): The current state of the menu.
        cursorBFS (Tuple[int, int, int]): Color of the BFS cursor.
        cursorDFS (Tuple[int, int, int]): Color of the DFS cursor.
        cursorASTAR (Tuple[int, int, int]): Color of the AStar cursor.
        cursorGA (Tuple[int, int, int]): Color of the GA cursor.
        BFSx (int): X-coordinate for BFS option.
        BFSy (int): Y-coordinate for BFS option.
        DFSx (int): X-coordinate for DFS option.
        DFSy (int): Y-coordinate for DFS option.
        ASTARx (int): X-coordinate for AStar option.
        ASTARy (int): Y-coordinate for AStar option.
        GAx (int): X-coordinate for GA option.
        GAy (int): Y-coordinate for GA option.
    """

    def __init__(self, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the main menu object.

        Parameters:
            game (Game): The game instance.
        """
        logging.debug("Initializing MainMenu class with game instance.")
        super().__init__(game)
        self.state: str = "BFS"
        self.cursorBFS: Tuple[int, int, int] = MENU_COLOR
        self.cursorDFS: Tuple[int, int, int] = WHITE
        self.cursorASTAR: Tuple[int, int, int] = WHITE
        self.cursorGA: Tuple[int, int, int] = WHITE
        self.BFSx, self.BFSy = self.mid_size, self.mid_size - 50
        self.DFSx, self.DFSy = self.mid_size, self.mid_size + 0
        self.ASTARx, self.ASTARy = self.mid_size, self.mid_size + 50
        self.GAx, self.GAy = self.mid_size, self.mid_size + 100
        self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)
        logging.debug(
            "MainMenu class initialized with state: {self.state}, cursor colors set, and cursor positions defined."
        )

    def change_cursor_color(self) -> None:
        """
        Changes the cursor color based on the current state.
        """
        logging.debug(f"Changing cursor color based on state: {self.state}.")
        try:
            self.clear_cursor_color()
            if self.state == "BFS":
                self.cursorBFS = MENU_COLOR
            elif self.state == "DFS":
                self.cursorDFS = MENU_COLOR
            elif self.state == "ASTAR":
                self.cursorASTAR = MENU_COLOR
            elif self.state == "GA":
                self.cursorGA = MENU_COLOR
            logging.debug("Cursor color changed to reflect current state.")
        except Exception as e:
            logging.error(f"Failed to change cursor color due to: {e}", exc_info=True)
            raise RuntimeError(f"Changing cursor color failed due to: {e}") from e

    def clear_cursor_color(self) -> None:
        """
        Resets the cursor colors to their default values.
        """
        logging.debug("Clearing cursor colors to default.")
        self.cursorBFS = WHITE
        self.cursorDFS = WHITE
        self.cursorASTAR = WHITE
        self.cursorGA = WHITE
        logging.debug("Cursor colors reset to default.")

    def display_menu(self) -> None:
        """
        Displays the main menu and handles its functionality.
        """
        logging.debug("Displaying main menu.")
        self.run_display = True
        try:
            while self.run_display:
                self.game.event_handler()
                self.check_input()
                self.game.display.fill(WINDOW_COLOR)

                self.game.draw_text(
                    "Ai Snake Game",
                    size=self.title_size,
                    x=self.game.SIZE / 2,
                    y=self.game.SIZE / 2 - 2 * (CELL_SIZE + NO_OF_CELLS),
                    color=TITLE_COLOR,
                )
                self.game.draw_text(
                    "BFS",
                    size=self.option_size,
                    x=self.BFSx,
                    y=self.BFSy,
                    color=self.cursorBFS,
                )
                self.game.draw_text(
                    "DFS",
                    size=self.option_size,
                    x=self.DFSx,
                    y=self.DFSy,
                    color=self.cursorDFS,
                )
                self.game.draw_text(
                    "AStar",
                    size=self.option_size,
                    x=self.ASTARx,
                    y=self.ASTARy,
                    color=self.cursorASTAR,
                )
                self.game.draw_text(
                    "Genetic Algorithm",
                    size=self.option_size,
                    x=self.GAx,
                    y=self.GAy,
                    color=self.cursorGA,
                )
                self.draw_cursor()
                self.change_cursor_color()
                self.blit_menu()
            logging.debug("Main menu displayed and interactive.")
        except Exception as e:
            logging.error(f"Failed to display main menu due to: {e}", exc_info=True)
            raise RuntimeError(f"Displaying main menu failed due to: {e}") from e

    def check_input(self) -> None:
        """
        Checks user input for menu navigation.
        """
        logging.debug("Checking user input for menu navigation.")
        try:
            self.move_cursor()
            if self.game.START:
                if self.state == "GA":  # go to genetic algorithm options
                    self.game.curr_menu = self.game.GA
                else:
                    self.game.playing = True
                self.run_display = False
                logging.debug("Input processed and state updated.")
        except Exception as e:
            logging.error(f"Failed to check input due to: {e}", exc_info=True)
            raise RuntimeError(f"Checking input failed due to: {e}") from e

    def move_cursor(self) -> None:
        """
        Moves the cursor based on user input.
        """
        logging.debug("Attempting to move cursor based on user input.")
        try:
            if self.game.DOWNKEY:
                if self.state == "BFS":
                    self.cursor_rect.midtop = (self.DFSx + self.offset, self.DFSy)
                    self.state = "DFS"
                    logging.debug("Cursor moved to DFS.")
                elif self.state == "DFS":
                    self.cursor_rect.midtop = (self.ASTARx + self.offset, self.ASTARy)
                    self.state = "ASTAR"
                    logging.debug("Cursor moved to AStar.")
                elif self.state == "ASTAR":
                    self.cursor_rect.midtop = (self.GAx + self.offset, self.GAy)
                    self.state = "GA"
                    logging.debug("Cursor moved to Genetic Algorithm.")
                elif self.state == "GA":
                    self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)
                    self.state = "BFS"
                    logging.debug("Cursor moved to BFS.")

            if self.game.UPKEY:
                if self.state == "BFS":
                    self.cursor_rect.midtop = (self.GAx + self.offset, self.GAy)
                    self.state = "GA"
                    logging.debug("Cursor moved up from BFS to Genetic Algorithm.")
                elif self.state == "DFS":
                    self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)
                    self.state = "BFS"
                    logging.debug("Cursor moved up from DFS to BFS.")
                elif self.state == "ASTAR":
                    self.cursor_rect.midtop = (self.DFSx + self.offset, self.DFSy)
                    self.state = "DFS"
                    logging.debug("Cursor moved up from AStar to DFS.")
                elif self.state == "GA":
                    self.cursor_rect.midtop = (self.ASTARx + self.offset, self.ASTARy)
                    self.state = "ASTAR"
                    logging.debug("Cursor moved up from Genetic Algorithm to AStar.")
        except Exception as e:
            logging.error(f"Failed to move cursor due to: {e}", exc_info=True)
            raise RuntimeError(f"Moving cursor failed due to: {e}") from e


class button:
    """
    A class to represent a button in the game interface.

    Attributes:
        x (int): The x-coordinate of the button.
        y (int): The y-coordinate of the button.
        text (str): The text displayed on the button.
        game (Game): The game instance.
        font (pygame.font.Font): The font used for the button text.
        clicked (bool): A flag to determine if the button has been clicked.
    """

    def __init__(self, x: int, y: int, text: str, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the button object.

        Parameters:
            x (int): The x-coordinate of the button.
            y (int): The y-coordinate of the button.
            text (str): The text displayed on the button.
            game (Game): The game instance.
        """
        logging.debug(f"Initializing button with text: {text}.")
        self.x: int = x
        self.y: int = y
        self.text: str = text
        self.game: "Game" = game
        self.font: pygame.font.Font = pygame.font.Font(game.font_name, 30)
        self.clicked: bool = False
        logging.debug(
            f"Button initialized at position ({self.x}, {self.y}) with text: {self.text}."
        )

    def draw_button(self) -> bool:
        """
        Draws the button and checks for interaction.

        Returns:
            bool: True if the button was clicked, False otherwise.
        """
        logging.debug("Attempting to draw button.")
        action: bool = False

        # get mouse position
        pos: Tuple[int, int] = pygame.mouse.get_pos()

        # create pygame Rect object for the button
        button_rect: pygame.Rect = pygame.Rect(self.x, self.y, BTN_WIDTH, BTN_HEIGHT)

        # check mouseover and clicked conditions
        if button_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                self.clicked = True
                pygame.draw.rect(self.game.display, BTN_CLICKED, button_rect)
                logging.debug("Button clicked.")
            elif pygame.mouse.get_pressed()[0] == 0 and self.clicked == True:
                self.clicked = False
                action = True
                logging.debug("Button action triggered.")
            else:
                pygame.draw.rect(self.game.display, BTN_HOVER, button_rect)
                logging.debug("Mouse hovered over button.")
        else:
            pygame.draw.rect(self.game.display, BTN_COLOR, button_rect)
            logging.debug("Button drawn with default color.")

        # add text to button
        text_img: pygame.Surface = self.font.render(self.text, True, WHITE)
        text_len: int = text_img.get_width()
        self.game.display.blit(
            text_img, (self.x + int(BTN_WIDTH / 2) - int(text_len / 2), self.y + 25)
        )
        logging.debug("Text added to button.")

        return action


class TextBox:
    """
    A class to represent a text input box in the game interface.

    Attributes:
        x (int): The x-coordinate of the text box.
        y (int): The y-coordinate of the text box.
        game (Game): The game instance.
        font (pygame.font.Font): The font used for the text input.
        input_rect (pygame.Rect): The rectangle representing the text box.
        input (str): The text input by the user.
        active (bool): A flag to determine if the text box is active.
    """

    def __init__(self, x: int, y: int, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the text box object.

        Parameters:
            x (int): The x-coordinate of the text box.
            y (int): The y-coordinate of the text box.
            game (Game): The game instance.
        """
        logging.debug("Initializing TextBox.")
        self.font: pygame.font.Font = pygame.font.Font(game.font_name, 20)
        self.input_rect: pygame.Rect = pygame.Rect(x, y, TXT_WIDTH, TXT_HEIGHT)
        self.input: str = ""
        self.game: "Game" = game
        self.active: bool = False
        logging.debug(f"TextBox initialized at position ({x}, {y}).")

    def draw_input(self) -> None:
        """
        Draws the text input box and handles interaction.
        """
        logging.debug("Attempting to draw input box.")
        # get mouse position
        pos: Tuple[int, int] = pygame.mouse.get_pos()

        if self.input_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                self.active = True
                logging.debug("TextBox activated.")

        elif pygame.mouse.get_pressed()[0] == 1:
            self.active = False
            logging.debug("TextBox deactivated.")

        if self.active:
            color: Tuple[int, int, int] = TXT_ACTIVE
        else:
            color: Tuple[int, int, int] = TXT_PASSIVE

        pygame.draw.rect(self.game.display, color, self.input_rect, 2)
        text_surface: pygame.Surface = self.font.render(self.input, False, WHITE)
        self.game.display.blit(
            text_surface, (self.input_rect.x + 15, self.input_rect.y + 1)
        )
        logging.debug("Input drawn in TextBox.")


class GAMenu(Menu):
    """
    A class to represent the Genetic Algorithm menu, inheriting from Menu.

    Attributes:
        controller (Controller): The GA controller.
        train_model (button): The button to start training the model.
        load_model (button): The button to load a pre-trained model.
        no_population (TextBox): The text box for number of populations.
        no_generation (TextBox): The text box for number of generations.
        no_hidden_nodes (TextBox): The text box for number of hidden nodes.
        mutation_rate (TextBox): The text box for mutation rate.
    """

    def __init__(self, game: "Game", controller: "Controller") -> None:
        """
        Constructs all the necessary attributes for the GA menu object.

        Parameters:
            game (Game): The game instance.
            controller (Controller): The GA controller.
        """
        logging.debug("Initializing GAMenu class with game instance and controller.")
        Menu.__init__(self, game)

        self.controller: "Controller" = controller
        self.train_model: button = button(
            game.SIZE / 2 - 4 * (CELL_SIZE + NO_OF_CELLS),
            game.SIZE / 2 + 3.5 * (CELL_SIZE + NO_OF_CELLS),
            "Train Model",
            game,
        )
        self.load_model: button = button(
            game.SIZE / 2 + (CELL_SIZE),
            game.SIZE / 2 + 3.5 * (CELL_SIZE + NO_OF_CELLS),
            "Load Model",
            game,
        )

        self.no_population: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 - 60, game
        )
        self.no_generation: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 - 10, game
        )
        self.no_hidden_nodes: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 + 40, game
        )
        self.mutation_rate: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 + 90, game
        )
        self.init_input()

    def init_input(self) -> None:
        """
        Initializes the input values for the GA menu.
        """
        logging.debug("Initializing input values for GAMenu.")
        self.no_population.input = "300"
        self.no_generation.input = "30"
        self.no_hidden_nodes.input = "8"
        self.mutation_rate.input = "12"
        logging.debug(
            "Input values initialized: Population=300, Generation=30, Hidden Nodes=8, Mutation Rate=12%."
        )

    def display_menu(self) -> None:
        """
        Displays the GA menu and handles its functionality.
        """
        logging.debug("Displaying GA menu.")
        self.run_display = True
        while self.run_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.running, self.game.playing = False, False
                    self.game.curr_menu.run_display = False
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.game.BACK = True

                    if self.no_population.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.no_population.input = self.no_population.input[:-1]
                        else:
                            if (
                                event.unicode.isdigit()
                                and len(self.no_population.input) < 3
                            ):
                                self.no_population.input += event.unicode

                    if self.no_generation.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.no_generation.input = self.no_generation.input[:-1]
                        else:
                            if (
                                event.unicode.isdigit()
                                and len(self.no_generation.input) < 3
                            ):
                                self.no_generation.input += event.unicode

                    if self.no_hidden_nodes.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.no_hidden_nodes.input = self.no_hidden_nodes.input[:-1]
                        else:
                            if (
                                event.unicode.isdigit()
                                and len(self.no_hidden_nodes.input) < 2
                            ):
                                self.no_hidden_nodes.input += event.unicode

                    if self.mutation_rate.active:
                        if event.key == pygame.K_BACKSPACE:
                            self.mutation_rate.input = self.mutation_rate.input[:-1]
                        else:
                            if (
                                event.unicode.isdigit()
                                and len(self.mutation_rate.input) < 3
                            ):
                                self.mutation_rate.input += event.unicode

            self.check_input()
            self.game.display.fill(WINDOW_COLOR)

            self.game.draw_text(
                "GA Options",
                self.title_size,
                self.game.SIZE / 2,
                self.game.SIZE / 2 - 4 * (CELL_SIZE + NO_OF_CELLS),
                color=TITLE_COLOR,
            )
            self.game.draw_text(
                "Settings to train model:",
                25,
                self.game.SIZE / 2,
                self.game.SIZE / 2 - 2 * (CELL_SIZE + NO_OF_CELLS),
                color=MENU_COLOR,
            )
            self.game.draw_text(
                "No. of populations     : ",
                20,
                self.game.SIZE / 2 - 2 * CELL_SIZE,
                self.game.SIZE / 2 - 50,
                color=BANNER_COLOR,
            )

            self.game.draw_text(
                "No. of generations     : ",
                20,
                self.game.SIZE / 2 - 2 * CELL_SIZE,
                self.game.SIZE / 2,
                color=BANNER_COLOR,
            )

            self.game.draw_text(
                "No. of hidden nodes   : ",
                20,
                self.game.SIZE / 2 - 2 * CELL_SIZE,
                self.game.SIZE / 2 + 50,
                color=BANNER_COLOR,
            )

            self.game.draw_text(
                "Mutation rate %:          : ",
                20,
                self.game.SIZE / 2 - 2 * CELL_SIZE,
                self.game.SIZE / 2 + 100,
                color=BANNER_COLOR,
            )

            self.no_population.draw_input()
            self.no_generation.draw_input()
            self.no_hidden_nodes.draw_input()

            self.mutation_rate.draw_input()

            if self.load_model.draw_button():
                self.load_GA()
            if self.train_model.draw_button():
                self.train_GA()

            self.game.draw_text(
                "Q to return to main menu",
                20,
                self.game.SIZE / 2,
                self.game.SIZE / 2 + 6 * (NO_OF_CELLS + CELL_SIZE),
                color=WHITE,
            )

            self.blit_menu()
        self.reset()

    def reset(self):
        """
        Resets the active state of all text boxes.
        """
        self.no_population.active = False
        self.no_generation.active = False
        self.no_hidden_nodes.active = False
        self.mutation_rate.active = False

    def check_input(self):
        """
        Checks if the back key was pressed to return to the main menu.
        """
        if self.game.BACK:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False

    def load_GA(self):
        """
        Loads the GA model and switches to the main menu.
        """
        self.game.curr_menu = self.game.main_menu
        self.run_display = False
        self.game.curr_menu.state = "GA"
        self.game.playing = True
        self.game.load_model = True

    def train_GA(self):
        """
        Trains the GA model and switches to the main menu.
        """
        self.game.curr_menu = self.game.main_menu
        self.run_display = False
        self.game.curr_menu.state = "GA"
        self.game.playing = True

        if len(self.no_population.input) > 0:
            Population.population = int(self.no_population.input)

        if len(self.no_hidden_nodes.input) > 0:
            Population.hidden_node = int(self.no_hidden_nodes.input)

        if len(self.no_generation.input) > 0:
            GA.generation = int(self.no_generation.input)

        if len(self.mutation_rate.input) > 0:
            GA.mutation_rate = int(self.mutation_rate.input) / 100
