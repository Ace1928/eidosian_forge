from __future__ import annotations
import html
import itertools
import re
import unicodedata
def _build_html_entities():
    entities = {}
    for name, char in html.entities.html5.items():
        if name.endswith(';'):
            entities['&' + name] = char
            if name == name.lower():
                name_upper = name.upper()
                entity_upper = '&' + name_upper
                if html.unescape(entity_upper) == entity_upper:
                    entities[entity_upper] = char.upper()
    return entities