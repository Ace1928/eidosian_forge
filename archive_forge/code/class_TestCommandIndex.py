import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class TestCommandIndex(TestHelp):
    """Tests for the HelpCommandIndex class."""

    def test_default_constructable(self):
        index = commands.HelpCommandIndex()

    def test_get_topics_None(self):
        """Searching for None returns an empty list."""
        index = commands.HelpCommandIndex()
        self.assertEqual([], index.get_topics(None))

    def test_get_topics_rocks(self):
        """Searching for 'rocks' returns the cmd_rocks command instance."""
        index = commands.HelpCommandIndex()
        topics = index.get_topics('rocks')
        self.assertEqual(1, len(topics))
        self.assertIsInstance(topics[0], builtins.cmd_rocks)

    def test_get_topics_no_topic(self):
        """Searching for something that is not a command returns []."""
        index = commands.HelpCommandIndex()
        self.assertEqual([], index.get_topics('nothing by this name'))

    def test_prefix(self):
        """CommandIndex has a prefix of 'commands/'."""
        index = commands.HelpCommandIndex()
        self.assertEqual('commands/', index.prefix)

    def test_get_topic_with_prefix(self):
        """Searching for commands/rocks returns the rocks command object."""
        index = commands.HelpCommandIndex()
        topics = index.get_topics('commands/rocks')
        self.assertEqual(1, len(topics))
        self.assertIsInstance(topics[0], builtins.cmd_rocks)