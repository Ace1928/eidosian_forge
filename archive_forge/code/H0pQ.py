from ai_logic import (
    expectimax,
    simulate_move,
    is_game_over,
    calculate_best_move,
)
from gui_utils import (
    update_gui,
    get_tile_color,
    get_tile_text,
    get_tile_font_size,
    get_tile_font_color,
    get_tile_font_weight,
    get_tile_font_family,
)
import types
import importlib.util
import logging


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
def main_game_loop():
    board = initialize_game()
    score = 0
    game_over = False

    while not game_over:
        best_move = calculate_best_move(board)
        if best_move:
            board, move_score = simulate_move(board, best_move)
            score += move_score
            add_random_tile(board)
            update_gui(board, score)
            game_over = is_game_over(board)
        else:
            game_over = True

    print(f"Game Over! Final Score: {score}")
