import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
class TestModuleHelpTopic(tests.TestCase):
    """Tests for the ModuleHelpTopic class."""

    def test_contruct(self):
        """Construction takes the module to document."""
        mod = FakeModule('foo', 'foo')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual(mod, topic.module)

    def test_get_help_text_None(self):
        """A ModuleHelpTopic returns the docstring for get_help_text."""
        mod = FakeModule(None, 'demo')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual("Plugin 'demo' has no docstring.\n", topic.get_help_text())

    def test_get_help_text_no_carriage_return(self):
        """ModuleHelpTopic.get_help_text adds a 
 if needed."""
        mod = FakeModule('one line of help', 'demo')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual('one line of help\n', topic.get_help_text())

    def test_get_help_text_carriage_return(self):
        """ModuleHelpTopic.get_help_text adds a 
 if needed."""
        mod = FakeModule('two lines of help\nand more\n', 'demo')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual('two lines of help\nand more\n', topic.get_help_text())

    def test_get_help_text_with_additional_see_also(self):
        mod = FakeModule('two lines of help\nand more', 'demo')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual('two lines of help\nand more\n\n:See also: bar, foo\n', topic.get_help_text(['foo', 'bar']))

    def test_get_help_topic(self):
        """The help topic for a plugin is its module name."""
        mod = FakeModule('two lines of help\nand more', 'breezy.plugins.demo')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual('demo', topic.get_help_topic())
        mod = FakeModule('two lines of help\nand more', 'breezy.plugins.foo_bar')
        topic = plugin.ModuleHelpTopic(mod)
        self.assertEqual('foo_bar', topic.get_help_topic())