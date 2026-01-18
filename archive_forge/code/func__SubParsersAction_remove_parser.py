from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _SubParsersAction_remove_parser(self: argparse._SubParsersAction, name: str) -> None:
    """
    Removes a sub-parser from a sub-parsers group. Used to remove subcommands from a parser.

    This function is added by cmd2 as a method called ``remove_parser()`` to ``argparse._SubParsersAction`` class.

    To call: ``action.remove_parser(name)``

    :param self: instance of the _SubParsersAction being edited
    :param name: name of the subcommand for the sub-parser to remove
    """
    for choice_action in self._choices_actions:
        if choice_action.dest == name:
            self._choices_actions.remove(choice_action)
            break
    subparser = self._name_parser_map.get(name)
    if subparser is not None:
        to_remove = []
        for cur_name, cur_parser in self._name_parser_map.items():
            if cur_parser is subparser:
                to_remove.append(cur_name)
        for cur_name in to_remove:
            del self._name_parser_map[cur_name]