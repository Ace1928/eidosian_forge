import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class TestTopicIndex(TestHelp):
    """Tests for the HelpTopicIndex class."""

    def test_default_constructable(self):
        index = help_topics.HelpTopicIndex()

    def test_get_topics_None(self):
        """Searching for None returns the basic help topic."""
        index = help_topics.HelpTopicIndex()
        topics = index.get_topics(None)
        self.assertEqual(1, len(topics))
        self.assertIsInstance(topics[0], help_topics.RegisteredTopic)
        self.assertEqual('basic', topics[0].topic)

    def test_get_topics_topics(self):
        """Searching for a string returns the matching string."""
        index = help_topics.HelpTopicIndex()
        topics = index.get_topics('topics')
        self.assertEqual(1, len(topics))
        self.assertIsInstance(topics[0], help_topics.RegisteredTopic)
        self.assertEqual('topics', topics[0].topic)

    def test_get_topics_no_topic(self):
        """Searching for something not registered returns []."""
        index = help_topics.HelpTopicIndex()
        self.assertEqual([], index.get_topics('nothing by this name'))

    def test_prefix(self):
        """TopicIndex has a prefix of ''."""
        index = help_topics.HelpTopicIndex()
        self.assertEqual('', index.prefix)