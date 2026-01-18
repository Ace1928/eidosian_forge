import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
class TestEntitySubstitution(object):
    """Standalone tests of the EntitySubstitution class."""

    def setup_method(self):
        self.sub = EntitySubstitution

    @pytest.mark.parametrize('original,substituted', [('foo∀☃õbar', 'foo&forall;☃&otilde;bar'), ('‘’foo“”', '&lsquo;&rsquo;foo&ldquo;&rdquo;')])
    def test_substitute_html(self, original, substituted):
        assert self.sub.substitute_html(original) == substituted

    def test_html5_entity(self):
        for entity, u in (('&models;', '⊧'), ('&Nfr;', '𝔑'), ('&ngeqq;', '≧̸'), ('&not;', '¬'), ('&Not;', '⫬'), '||', ('fj', 'fj'), ('&gt;', '>'), ('&lt;', '<'), ('&amp;', '&')):
            template = '3 %s 4'
            raw = template % u
            with_entities = template % entity
            assert self.sub.substitute_html(raw) == with_entities

    def test_html5_entity_with_variation_selector(self):
        data = 'fjords ⊔ penguins'
        markup = 'fjords &sqcup; penguins'
        assert self.sub.substitute_html(data) == markup
        data = 'fjords ⊔︀ penguins'
        markup = 'fjords &sqcups; penguins'
        assert self.sub.substitute_html(data) == markup

    def test_xml_converstion_includes_no_quotes_if_make_quoted_attribute_is_false(self):
        s = 'Welcome to "my bar"'
        assert self.sub.substitute_xml(s, False) == s

    def test_xml_attribute_quoting_normally_uses_double_quotes(self):
        assert self.sub.substitute_xml('Welcome', True) == '"Welcome"'
        assert self.sub.substitute_xml("Bob's Bar", True) == '"Bob\'s Bar"'

    def test_xml_attribute_quoting_uses_single_quotes_when_value_contains_double_quotes(self):
        s = 'Welcome to "my bar"'
        assert self.sub.substitute_xml(s, True) == '\'Welcome to "my bar"\''

    def test_xml_attribute_quoting_escapes_single_quotes_when_value_contains_both_single_and_double_quotes(self):
        s = 'Welcome to "Bob\'s Bar"'
        assert self.sub.substitute_xml(s, True) == '"Welcome to &quot;Bob\'s Bar&quot;"'

    def test_xml_quotes_arent_escaped_when_value_is_not_being_quoted(self):
        quoted = 'Welcome to "Bob\'s Bar"'
        assert self.sub.substitute_xml(quoted) == quoted

    def test_xml_quoting_handles_angle_brackets(self):
        assert self.sub.substitute_xml('foo<bar>') == 'foo&lt;bar&gt;'

    def test_xml_quoting_handles_ampersands(self):
        assert self.sub.substitute_xml('AT&T') == 'AT&amp;T'

    def test_xml_quoting_including_ampersands_when_they_are_part_of_an_entity(self):
        assert self.sub.substitute_xml('&Aacute;T&T') == '&amp;Aacute;T&amp;T'

    def test_xml_quoting_ignoring_ampersands_when_they_are_part_of_an_entity(self):
        assert self.sub.substitute_xml_containing_entities('&Aacute;T&T') == '&Aacute;T&amp;T'

    def test_quotes_not_html_substituted(self):
        """There's no need to do this except inside attribute values."""
        text = 'Bob\'s "bar"'
        assert self.sub.substitute_html(text) == text