import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _gtk4_setup_actions(self) -> None:
    action_group = Gio.SimpleActionGroup.new()
    action = Gio.SimpleAction.new('ignore-all', GLib.VariantType('s'))
    action.connect('activate', lambda _action, word: self.ignore_all(word.get_string()))
    action_group.add_action(action)
    action = Gio.SimpleAction.new('add-to-dictionary', GLib.VariantType('s'))
    action.connect('activate', lambda _action, word: self.add_to_dictionary(word.get_string()))
    action_group.add_action(action)
    action = Gio.SimpleAction.new('replace-word', GLib.VariantType('s'))
    action.connect('activate', lambda _action, suggestion: self._replace_word(suggestion.get_string()))
    action_group.add_action(action)
    language = Gio.PropertyAction.new('language', self, 'language')
    action_group.add_action(language)
    self._view.insert_action_group('spelling', action_group)