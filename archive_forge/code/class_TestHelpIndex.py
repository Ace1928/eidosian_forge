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
class TestHelpIndex(tests.TestCase):
    """Tests for the PluginsHelpIndex class."""

    def test_default_constructable(self):
        index = plugin.PluginsHelpIndex()

    def test_get_topics_None(self):
        """Searching for None returns an empty list."""
        index = plugin.PluginsHelpIndex()
        self.assertEqual([], index.get_topics(None))

    def test_get_topics_for_plugin(self):
        """Searching for plugin name gets its docstring."""
        index = plugin.PluginsHelpIndex()
        self.assertFalse('breezy.plugins.demo_module' in sys.modules)
        demo_module = FakeModule('', 'breezy.plugins.demo_module')
        sys.modules['breezy.plugins.demo_module'] = demo_module
        try:
            topics = index.get_topics('demo_module')
            self.assertEqual(1, len(topics))
            self.assertIsInstance(topics[0], plugin.ModuleHelpTopic)
            self.assertEqual(demo_module, topics[0].module)
        finally:
            del sys.modules['breezy.plugins.demo_module']

    def test_get_topics_no_topic(self):
        """Searching for something that is not a plugin returns []."""
        index = plugin.PluginsHelpIndex()
        self.assertEqual([], index.get_topics('nothing by this name'))

    def test_prefix(self):
        """PluginsHelpIndex has a prefix of 'plugins/'."""
        index = plugin.PluginsHelpIndex()
        self.assertEqual('plugins/', index.prefix)

    def test_get_plugin_topic_with_prefix(self):
        """Searching for plugins/demo_module returns help."""
        index = plugin.PluginsHelpIndex()
        self.assertFalse('breezy.plugins.demo_module' in sys.modules)
        demo_module = FakeModule('', 'breezy.plugins.demo_module')
        sys.modules['breezy.plugins.demo_module'] = demo_module
        try:
            topics = index.get_topics('plugins/demo_module')
            self.assertEqual(1, len(topics))
            self.assertIsInstance(topics[0], plugin.ModuleHelpTopic)
            self.assertEqual(demo_module, topics[0].module)
        finally:
            del sys.modules['breezy.plugins.demo_module']