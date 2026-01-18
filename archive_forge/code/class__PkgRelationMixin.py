import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class _PkgRelationMixin(object):
    """Package relationship mixin

    Inheriting from this mixin you can extend a :class:`Deb822` object with
    attributes letting you access inter-package relationship in a structured
    way, rather than as strings.
    For example, while you can usually use ``pkg['depends']`` to
    obtain the Depends string of package pkg, mixing in with this class you
    gain pkg.depends to access Depends as a Pkgrel instance

    To use, subclass _PkgRelationMixin from a class with a _relationship_fields
    attribute. It should be a list of field names for which structured access
    is desired; for each of them a method wild be added to the inherited class.
    The method name will be the lowercase version of field name; '-' will be
    mangled as '_'. The method would return relationships in the same format of
    the PkgRelation' relations property.

    See Packages and Sources as examples.
    """
    _relationship_fields = []

    def __init__(self, *args, **kwargs):
        self.__relations = _lowercase_dict({})
        self.__parsed_relations = False
        for name in self._relationship_fields:
            keyname = name.lower()
            if name in self:
                self.__relations[keyname] = None
            else:
                self.__relations[keyname] = []

    @property
    def relations(self):
        """Return a dictionary of inter-package relationships among the current
        and other packages.

        Dictionary keys depend on the package kind. Binary packages have keys
        like 'depends', 'recommends', ... while source packages have keys like
        'build-depends', 'build-depends-indep' and so on. See the Debian policy
        for the comprehensive field list.

        Dictionary values are package relationships returned as lists of lists
        of dictionaries (see below for some examples).

        The encoding of package relationships is as follows:

        - the top-level lists corresponds to the comma-separated list of
          :class:`Deb822`, their components form a conjunction, i.e. they
          have to be AND-ed together
        - the inner lists corresponds to the pipe-separated list of
          :class:`Deb822`,
          their components form a disjunction, i.e. they have to be OR-ed
          together
        - member of the inner lists are dictionaries with the following keys:

          ``name``
            package (or virtual package) name
          ``version``
            A pair <`operator`, `version`> if the relationship is
            versioned, None otherwise. operator is one of ``<<``,
            ``<=``, ``=``, ``>=``, ``>>``; version is the given version as
            a string.
          ``arch``
            A list of pairs <`enabled`, `arch`> if the
            relationship is architecture specific, None otherwise.
            Enabled is a boolean (``False`` if the architecture is
            negated with ``!``, ``True`` otherwise), arch the
            Debian architecture name as a string.
          ``restrictions``
            A list of lists of tuples <`enabled`, `profile`>
            if there is a restriction formula defined, ``None``
            otherwise. Each list of tuples represents a restriction
            list while each tuple represents an individual term
            within the restriction list. Enabled is a boolean
            (``False`` if the restriction is negated with ``!``,
            ``True`` otherwise). The profile is the name of the
            build restriction.
            https://wiki.debian.org/BuildProfileSpec

          The arch and restrictions tuples are available as named tuples so
          elements are available as `term[0]` or alternatively as
          `term.enabled` (and so forth).

        Examples:

        ``"emacs | emacsen, make, debianutils (>= 1.7)"``
        becomes::

          [
            [ {'name': 'emacs'}, {'name': 'emacsen'} ],
            [ {'name': 'make'} ],
            [ {'name': 'debianutils', 'version': ('>=', '1.7')} ]
          ]

        ``"tcl8.4-dev, procps [!hurd-i386]"``
        becomes::

          [
            [ {'name': 'tcl8.4-dev'} ],
            [ {'name': 'procps', 'arch': (false, 'hurd-i386')} ]
          ]

        ``"texlive <!cross>"``
        becomes::

          [
            [ {'name': 'texlive', 'restriction': [[(false, 'cross')]]} ]
          ]
        """
        if not self.__parsed_relations:
            lazy_rels = filter(lambda n: self.__relations[n] is None, self.__relations.keys())
            for n in lazy_rels:
                self.__relations[n] = PkgRelation.parse_relations(self[n])
            self.__parsed_relations = True
        return self.__relations