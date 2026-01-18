from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
Given a list of supported values, finds the best match from
        the list of accepted values.

        Language tags are normalized for the purpose of matching, but
        are returned unchanged.

        If no exact match is found, this will fall back to matching
        the first subtag (primary language only), first with the
        accepted values then with the match values. This partial is not
        applied to any other language subtags.

        The default is returned if no exact or fallback match is found.

        :param matches: A list of supported languages to find a match.
        :param default: The value that is returned if none match.
        