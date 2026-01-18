import __main__
from collections import namedtuple
import logging
import traceback
import re
import os
import sys
from jedi import Interpreter
class JediRL:

    def complete(self, text, state):
        """
            This complete stuff is pretty weird, a generator would make
            a lot more sense, but probably due to backwards compatibility
            this is still the way how it works.

            The only important part is stuff in the ``state == 0`` flow,
            everything else has been copied from the ``rlcompleter`` std.
            library module.
            """
        if state == 0:
            sys.path.insert(0, os.getcwd())
            try:
                logging.debug('Start REPL completion: ' + repr(text))
                interpreter = Interpreter(text, [namespace_module.__dict__])
                completions = interpreter.complete(fuzzy=fuzzy)
                logging.debug('REPL completions: %s', completions)
                self.matches = [text[:len(text) - c._like_name_length] + c.name_with_symbols for c in completions]
            except:
                logging.error('REPL Completion error:\n' + traceback.format_exc())
                raise
            finally:
                sys.path.pop(0)
        try:
            return self.matches[state]
        except IndexError:
            return None