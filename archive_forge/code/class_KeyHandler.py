from typing import List
from .keymap import KEYMAP, get_character
class KeyHandler(type):
    """
    Metaclass that adds the key handlers to the class
    """

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if not hasattr(new_cls, 'key_handler'):
            new_cls.key_handler = {}
        new_cls.handle_input = KeyHandler.handle_input
        for value in attrs.values():
            handled_keys = getattr(value, 'handle_key', [])
            for key in handled_keys:
                new_cls.key_handler[key] = value
        return new_cls

    @staticmethod
    def handle_input(cls):
        """Finds and returns the selected character if it exists in the handler"""
        char = get_character()
        if char != KEYMAP['undefined']:
            char = ord(char)
        handler = cls.key_handler.get(char)
        if handler:
            cls.current_selection = char
            return handler(cls)
        else:
            return None