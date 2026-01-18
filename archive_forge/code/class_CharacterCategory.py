from enum import Enum, Flag
class CharacterCategory:
    """
    This enum represents the different categories language models for
    ``SingleByteCharsetProber`` put characters into.

    Anything less than CONTROL is considered a letter.
    """
    UNDEFINED = 255
    LINE_BREAK = 254
    SYMBOL = 253
    DIGIT = 252
    CONTROL = 251