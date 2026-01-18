from typing import List, Tuple
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
def get_tile_color(value: int) -> tuple:
    """
    Generates a color for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        tuple: The color (R, G, B) for the tile.
    """
    # Example logic for generating a color based on the tile's value
    # This is a placeholder and should be replaced with a dynamic gradient generator
    base_color = (238, 228, 218)
    factor = log(value, 2) * 20  # Simplified example for color variation
    return tuple(min(max(0, x + int(factor)), 255) for x in base_color)
