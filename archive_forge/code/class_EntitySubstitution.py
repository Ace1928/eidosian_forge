from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
class EntitySubstitution(object):
    """The ability to substitute XML or HTML entities for certain characters."""

    def _populate_class_variables():
        """Initialize variables used by this class to manage the plethora of
        HTML5 named entities.

        This function returns a 3-tuple containing two dictionaries
        and a regular expression:

        unicode_to_name - A mapping of Unicode strings like "⦨" to
        entity names like "angmsdaa". When a single Unicode string has
        multiple entity names, we try to choose the most commonly-used
        name.

        name_to_unicode: A mapping of entity names like "angmsdaa" to 
        Unicode strings like "⦨".

        named_entity_re: A regular expression matching (almost) any
        Unicode string that corresponds to an HTML5 named entity.
        """
        unicode_to_name = {}
        name_to_unicode = {}
        short_entities = set()
        long_entities_by_first_character = defaultdict(set)
        for name_with_semicolon, character in sorted(html5.items()):
            if name_with_semicolon.endswith(';'):
                name = name_with_semicolon[:-1]
            else:
                name = name_with_semicolon
            if name not in name_to_unicode:
                name_to_unicode[name] = character
            unicode_to_name[character] = name
            if len(character) == 1 and ord(character) < 128 and (character not in '<>&'):
                continue
            if len(character) > 1 and all((ord(x) < 128 for x in character)):
                continue
            if len(character) == 1:
                short_entities.add(character)
            else:
                long_entities_by_first_character[character[0]].add(character)
        particles = set()
        for short in short_entities:
            long_versions = long_entities_by_first_character[short]
            if not long_versions:
                particles.add(short)
            else:
                ignore = ''.join([x[1] for x in long_versions])
                particles.add('%s(?![%s])' % (short, ignore))
        for long_entities in list(long_entities_by_first_character.values()):
            for long_entity in long_entities:
                particles.add(long_entity)
        re_definition = '(%s)' % '|'.join(particles)
        for codepoint, name in list(codepoint2name.items()):
            character = chr(codepoint)
            unicode_to_name[character] = name
        return (unicode_to_name, name_to_unicode, re.compile(re_definition))
    CHARACTER_TO_HTML_ENTITY, HTML_ENTITY_TO_CHARACTER, CHARACTER_TO_HTML_ENTITY_RE = _populate_class_variables()
    CHARACTER_TO_XML_ENTITY = {"'": 'apos', '"': 'quot', '&': 'amp', '<': 'lt', '>': 'gt'}
    BARE_AMPERSAND_OR_BRACKET = re.compile('([<>]|&(?!#\\d+;|#x[0-9a-fA-F]+;|\\w+;))')
    AMPERSAND_OR_BRACKET = re.compile('([<>&])')

    @classmethod
    def _substitute_html_entity(cls, matchobj):
        """Used with a regular expression to substitute the
        appropriate HTML entity for a special character string."""
        entity = cls.CHARACTER_TO_HTML_ENTITY.get(matchobj.group(0))
        return '&%s;' % entity

    @classmethod
    def _substitute_xml_entity(cls, matchobj):
        """Used with a regular expression to substitute the
        appropriate XML entity for a special character string."""
        entity = cls.CHARACTER_TO_XML_ENTITY[matchobj.group(0)]
        return '&%s;' % entity

    @classmethod
    def quoted_attribute_value(self, value):
        """Make a value into a quoted XML attribute, possibly escaping it.

         Most strings will be quoted using double quotes.

          Bob's Bar -> "Bob's Bar"

         If a string contains double quotes, it will be quoted using
         single quotes.

          Welcome to "my bar" -> 'Welcome to "my bar"'

         If a string contains both single and double quotes, the
         double quotes will be escaped, and the string will be quoted
         using double quotes.

          Welcome to "Bob's Bar" -> "Welcome to &quot;Bob's bar&quot;
        """
        quote_with = '"'
        if '"' in value:
            if "'" in value:
                replace_with = '&quot;'
                value = value.replace('"', replace_with)
            else:
                quote_with = "'"
        return quote_with + value + quote_with

    @classmethod
    def substitute_xml(cls, value, make_quoted_attribute=False):
        """Substitute XML entities for special XML characters.

        :param value: A string to be substituted. The less-than sign
          will become &lt;, the greater-than sign will become &gt;,
          and any ampersands will become &amp;. If you want ampersands
          that appear to be part of an entity definition to be left
          alone, use substitute_xml_containing_entities() instead.

        :param make_quoted_attribute: If True, then the string will be
         quoted, as befits an attribute value.
        """
        value = cls.AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)
        if make_quoted_attribute:
            value = cls.quoted_attribute_value(value)
        return value

    @classmethod
    def substitute_xml_containing_entities(cls, value, make_quoted_attribute=False):
        """Substitute XML entities for special XML characters.

        :param value: A string to be substituted. The less-than sign will
          become &lt;, the greater-than sign will become &gt;, and any
          ampersands that are not part of an entity defition will
          become &amp;.

        :param make_quoted_attribute: If True, then the string will be
         quoted, as befits an attribute value.
        """
        value = cls.BARE_AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)
        if make_quoted_attribute:
            value = cls.quoted_attribute_value(value)
        return value

    @classmethod
    def substitute_html(cls, s):
        """Replace certain Unicode characters with named HTML entities.

        This differs from data.encode(encoding, 'xmlcharrefreplace')
        in that the goal is to make the result more readable (to those
        with ASCII displays) rather than to recover from
        errors. There's absolutely nothing wrong with a UTF-8 string
        containg a LATIN SMALL LETTER E WITH ACUTE, but replacing that
        character with "&eacute;" will make it more readable to some
        people.

        :param s: A Unicode string.
        """
        return cls.CHARACTER_TO_HTML_ENTITY_RE.sub(cls._substitute_html_entity, s)