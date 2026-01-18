import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class TestRegisteredTopic(TestHelp):
    """Tests for the RegisteredTopic class."""

    def test_contruct(self):
        """Construction takes the help topic name for the registered item."""
        self.assertTrue('basic' in help_topics.topic_registry)
        topic = help_topics.RegisteredTopic('basic')
        self.assertEqual('basic', topic.topic)

    def test_get_help_text(self):
        """RegisteredTopic returns the get_detail results for get_help_text."""
        topic = help_topics.RegisteredTopic('commands')
        self.assertEqual(help_topics.topic_registry.get_detail('commands'), topic.get_help_text())

    def test_get_help_text_with_additional_see_also(self):
        topic = help_topics.RegisteredTopic('commands')
        self.assertEndsWith(topic.get_help_text(['foo', 'bar']), '\nSee also: bar, foo\n')

    def test_get_help_text_loaded_from_file(self):
        topic = help_topics.RegisteredTopic('authentication')
        self.assertStartsWith(topic.get_help_text(), 'Authentication Settings\n=======================\n\n')

    def test_get_help_topic(self):
        """The help topic for RegisteredTopic is its topic from construction."""
        topic = help_topics.RegisteredTopic('foobar')
        self.assertEqual('foobar', topic.get_help_topic())
        topic = help_topics.RegisteredTopic('baz')
        self.assertEqual('baz', topic.get_help_topic())