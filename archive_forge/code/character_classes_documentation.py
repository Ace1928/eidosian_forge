import re
from itertools import chain
from sys import maxunicode
from collections import Counter
from typing import AbstractSet, Any, Iterator, MutableSet, Optional, Union
from .unicode_subsets import RegexError, UnicodeSubset, UNICODE_CATEGORIES, unicode_subset

    A set class to represent XML Schema/XQuery/XPath regex character class.

    :param charset: a string with formatted character set.
    :param xsd_version: the reference XSD version for syntax variants. Defaults to '1.0'.
    TODO: implement __ior__, __iand__, __ixor__ operators for a full mutable set class.
    