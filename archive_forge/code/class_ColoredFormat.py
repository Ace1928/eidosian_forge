import re
from string import Formatter
class ColoredFormat:

    def __init__(self, tokens, messages_color_tokens):
        self._tokens = tokens
        self._messages_color_tokens = messages_color_tokens

    def strip(self):
        return AnsiParser.strip(self._tokens)

    def colorize(self, ansi_level):
        return AnsiParser.colorize(self._tokens, ansi_level)

    def make_coloring_message(self, message, *, ansi_level, colored_message):
        messages = [message if color_tokens is None else AnsiParser.wrap(colored_message.tokens, ansi_level=ansi_level, color_tokens=color_tokens) for color_tokens in self._messages_color_tokens]
        coloring = ColoringMessage(message)
        coloring._messages = iter(messages)
        return coloring