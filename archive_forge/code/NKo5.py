from ai_logic import expectimax, simulate_move
from gui_utils import update_gui
import types
import importlib.util


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


@StandardDecorator()
def main_game_loop():
    board = initialize_game()
    score = 0
    game_over = False

    while not game_over:
        _, best_move = expectimax(board, depth=3, playerTurn=True)
        board, move_score = simulate_move(board, best_move)
        score += move_score
        update_gui(board, score)
        game_over = is_game_over(board)

        if not game_over:
            add_random_tile(board)
            update_gui(board, score)
