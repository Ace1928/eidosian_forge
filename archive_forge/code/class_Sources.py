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
class Sources(Dsc, _PkgRelationMixin):
    """Represent an APT source package list

    This class is a thin wrapper around the parsing of :class:`Deb822`,
    using the field parsing of :class:`_PkgRelationMixin`.
    """
    _relationship_fields = ['build-depends', 'build-depends-indep', 'build-depends-arch', 'build-conflicts', 'build-conflicts-indep', 'build-conflicts-arch', 'binary']

    def __init__(self, *args, **kwargs):
        Dsc.__init__(self, *args, **kwargs)
        _PkgRelationMixin.__init__(self, *args, **kwargs)

    @classmethod
    def iter_paragraphs(cls, sequence, fields=None, use_apt_pkg=True, shared_storage=False, encoding='utf-8', strict=None):
        """Generator that yields a Deb822 object for each paragraph in Sources.

        Note that this overloaded form of the generator uses apt_pkg (a strict
        but fast parser) by default.

        See the :func:`~Deb822.iter_paragraphs` function for details.
        """
        if not strict:
            strict = {'whitespace-separates-paragraphs': False}
        return super(Sources, cls).iter_paragraphs(sequence, fields, use_apt_pkg, shared_storage, encoding, strict)