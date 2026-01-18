from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
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