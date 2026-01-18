from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2


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
