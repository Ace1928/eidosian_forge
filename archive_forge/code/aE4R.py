from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
from game_manager import (
    initialize_game,
    add_random_tile,
    update_game_state,
    efficient_game_state_update,
    check_game_over,
    process_move,
    randomise_next_tile,
    setup_event_handlers,
    on_game_start,
    on_game_end,
    on_game_restart,
    on_game_exit,
    on_game_pause,
    on_game_resume,
    on_game_win,
    on_game_loss,
    on_game_update,
    on_game_move,
    on_game_event,
    on_game_input,
    on_game_output,
)


def import_from_path(name: str, path: str) -> types.ModuleType:
    """
    Dynamically imports a module from a given file path.

    Args:
        name (str): The name of the module.
        path (str): The file path to the module.

    Returns:
        types.ModuleType: The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


standard_decorator = import_from_path(
    "standard_decorator", "/home/lloyd/EVIE/standard_decorator.py"
)
StandardDecorator = standard_decorator.StandardDecorator
setup_logging = standard_decorator.setup_logging

setup_logging()


@StandardDecorator()
def get_tile_color(value: int) -> tuple:
    """
    Generates a color for the tile based on its value, using a gradient approach.

    Args:
        value (int): The value of the tile.

    Returns:
        tuple: The color (R, G, B) for the tile.
    """
    if value == 0:
        return (205, 193, 180)  # Color for empty tile
    base_log = log2(value)
    base_color = (
        255 - min(int(base_log * 20), 255),
        255 - min(int(base_log * 15), 255),
        220,
    )
    return base_color


@StandardDecorator()
def get_tile_text(value: int) -> str:
    """
    Generates the text to display on the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        str: The text to display on the tile.
    """
    if value == 0:
        return ""
    return str(value)


@StandardDecorator()
def get_tile_font_size(value: int) -> int:
    """
    Generates the font size for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        int: The font size for the tile.
    """
    if value < 100:
        return 55
    if value < 1000:
        return 45
    if value < 10000:
        return 35
    return 25


@StandardDecorator()
def get_tile_font_color(value: int) -> tuple:
    """
    Generates the font color for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        tuple: The font color (R, G, B) for the tile.
    """
    if value < 8:
        return (119, 110, 101)
    return (249, 246, 242)


@StandardDecorator()
def get_tile_font_weight(value: int) -> str:
    """
    Generates the font weight for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        str: The font weight for the tile.
    """
    if value < 100:
        return "bold"
    return "normal"


@StandardDecorator()
def get_tile_font_family(value: int) -> str:
    """
    Generates the font family for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        str: The font family for the tile.
    """
    return "Verdana" if value < 1000 else "Arial"


@StandardDecorator()
def update_gui(board: np.ndarray, score: int) -> None:
    """
    Updates the GUI with the current game state.

    Args:
        board (np.ndarray): The game board as a 2D NumPy array.
        score (int): The current score.
    """
    pass


def draw_board(board: np.ndarray) -> None:
    """
    Draws the game board to the console for visualization.

    Args:
        board (np.ndarray): The current game board.
    """
    for row in board:
        print(row)


def draw_gui(board: np.ndarray) -> None:
    """
    Draws the game board to the GUI for visualization.

    Args:
        board (np.ndarray): The current game board.
    """
    # GUI drawing logic goes here
    # This could involve updating the display with the current game state.


def draw_tile(tile_value: int) -> None:
    """
    Draws a tile with a specific value to the GUI for visualization.

    Args:
        tile_value (int): The value of the tile to be drawn.
    """
    # Tile drawing logic goes here
    # This could involve rendering a tile with the specified value on the display.


def draw_score(score: int) -> None:
    """
    Draws the current score to the GUI for visualization.

    Args:
        score (int): The current score.
    """
    # Score drawing logic goes here
    # This could involve updating the displayed score on the GUI.


def draw_game_over() -> None:
    """
    Draws the game over screen to the GUI for visualization.
    """
    # Game over screen drawing logic goes here
    # This could involve displaying a game over message and final score on the GUI.


def draw_win() -> None:
    """
    Draws the win screen to the GUI for visualization.
    """
    # Win screen drawing logic goes here
    # This could involve displaying a win message and final score on the GUI.


def draw_game_start() -> None:
    """
    Draws the game start screen to the GUI for visualization.
    """
    # Game start screen drawing logic goes here
    # This could involve displaying a start message and initial game state on the GUI.


def draw_game_restart() -> None:
    """
    Draws the game restart screen to the GUI for visualization.
    """
    # Game restart screen drawing logic goes here
    # This could involve displaying a restart message and initial game state on the GUI.


def draw_game_pause() -> None:
    """
    Draws the game pause screen to the GUI for visualization.
    """
    # Game pause screen drawing logic goes here
    # This could involve displaying a pause message and current game state on the GUI.


def draw_game_resume() -> None:
    """
    Draws the game resume screen to the GUI for visualization.
    """
    # Game resume screen drawing logic goes here
    # This could involve displaying a resume message and current game state on the GUI.


def draw_game_help() -> None:
    """
    Draws the game help screen to the GUI for visualization.
    """
    # Game help screen drawing logic goes here
    # This could involve displaying a help message and instructions on the GUI.


def draw_game_settings() -> None:
    """
    Draws the game settings screen to the GUI for visualization.
    """
    # Game settings screen drawing logic goes here
    # This could involve displaying a settings menu and options on the GUI.


def draw_game_exit() -> None:
    """
    Draws the game exit screen to the GUI for visualization.
    """
    # Game exit screen drawing logic goes here
    # This could involve displaying an exit confirmation message on the GUI.


def draw_game_save() -> None:
    """
    Draws the game save screen to the GUI for visualization.
    """
    # Game save screen drawing logic goes here
    # This could involve displaying a save confirmation message on the GUI.


def draw_game_load() -> None:
    """
    Draws the game load screen to the GUI for visualization.
    """
    # Game load screen drawing logic goes here
    # This could involve displaying a load confirmation message on the GUI.


def ui_design() -> None:
    """
    Designs the user interface for the game.

    This function is responsible for setting up the initial GUI layout and appearance.
    """
    # UI design logic goes here
    # This could involve creating the initial layout, colors, fonts, and styles for the GUI.


def draw_ui() -> None:
    """
    Draws the user interface for the game.

    This function is responsible for rendering the GUI elements on the screen.
    """
    # UI drawing logic goes here
    # This could involve rendering the GUI elements, buttons, labels, and other components on the screen.


def setup_ui_events() -> None:
    """
    Sets up event handling for the user interface.

    This function is responsible for defining how the GUI elements interact with the game logic.
    """
    # UI event setup logic goes here
    # This could involve defining event handlers for button clicks, key presses, and other user interactions.


def ui_interaction() -> None:
    """
    Handles user interaction with the game.

    This function is responsible for processing user input and updating the game state accordingly.
    """
    # UI interaction logic goes here
    # This could involve capturing user input, processing events, and updating the game state based on user actions.


def update_gui(board: np.ndarray, score: int) -> None:
    """
    Updates the GUI with the current game state.

    Args:
        board (np.ndarray): The game board as a 2D NumPy array.
        score (int): The current score.
    """
    draw_board(board)
    draw_gui(board)
    draw_score(score)


def main():
    board = np.zeros((4, 4), dtype=int)
    board[0, 0] = 2
    board[1, 1] = 4
    board[2, 2] = 8
    board[3, 3] = 16
    score = 0

    update_gui(board, score)
