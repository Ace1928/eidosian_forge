import numpy as np
from ai_logic import (
    calculate_best_move,
    simulate_move,
    is_game_over,
    expectimax,
    get_empty_tiles,
    calculate_smoothness_and_monotonicity,
    calculate_empty_tiles,
    calculate_max_tile,
    calculate_score,
    calculate_smoothness,
    calculate_monotonicity,
    heuristic_evaluation,
)
from gui_utils import (
    get_tile_color,
    get_tile_text,
    get_tile_font_size,
    get_tile_font_color,
    get_tile_font_weight,
    get_tile_font_family,
)
from typing import List, Tuple
import types
import importlib.util
import logging
import random


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
def initialize_game() -> np.ndarray:
    """
    Initializes the game board.

    Returns:
        np.ndarray: The initialized game board as a 2D NumPy array.
    """
    board = np.zeros((4, 4), dtype=int)
    # Add two initial tiles
    add_random_tile(board)
    add_random_tile(board)
    return board


@StandardDecorator()
def add_random_tile(board: np.ndarray) -> None:
    """
    Adds a random tile (2 or 4) to an empty position on the board.

    Args:
        board (np.ndarray): The game board.
    """
    empty_positions = list(zip(*np.where(board == 0)))
    if empty_positions:
        x, y = random.choice(empty_positions)
        board[x, y] = 2 if random.random() < 0.9 else 4


@StandardDecorator()
def update_game_state(board: np.ndarray, move: str) -> Tuple[np.ndarray, int]:
    """
    Updates the game state based on the move.

    Args:
        board (np.ndarray): The current game board.
        move (str): The move to be made ('up', 'down', 'left', 'right').

    Returns:
        Tuple[np.ndarray, int]: The updated game board and the score gained from the move.
    """
    new_board, score = simulate_move(board, move)
    add_random_tile(new_board)
    return new_board, score


# Additional functions for processing moves, checking game over conditions, etc., would also be defined here.
