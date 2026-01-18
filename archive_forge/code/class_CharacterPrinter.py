import sys
from typing import List, Tuple
from typing import List
from functools import wraps
from time import time
import logging
class CharacterPrinter:
    """
    A class for printing characters in a spectrum of colors.
    """

    def __init__(self, characters: List[str], colors: List[Tuple[int, int, int]]):
        """
        Initializes a CharacterPrinter instance with the provided characters and colors.

        :param characters: A list of characters to be printed.
        :param colors: A list of RGB color tuples.
        """
        self.characters = characters
        self.colors = colors

    def print_characters(self) -> None:
        """
        Prints each character in the list of characters in a series of colors.
        """
        for char in self.characters:
            print(f'Character: {char} - Unicode: {ord(char)}', end=' | ')
            for color in self.colors:
                print(f'\x1b[38;2;{color[0]};{color[1]};{color[2]}m{char}\x1b[0m', end=' ')
            print()