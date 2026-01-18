from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
class Protected(BaseMultipartText):
    """
    A :py:class:`Protected` represents a "protected" piece of text.

    - :py:meth:`Protected.lower`, :py:meth:`Protected.upper`,
      :py:meth:`Protected.capitalize`, and :py:meth:`Protected.capitalize()`
      are no-ops and just return the :py:class:`Protected` object itself.
    - :py:meth:`Protected.split` never splits the text. It always returns a
      one-element list containing the :py:class:`Protected` object itself.
    - In LaTeX output, :py:class:`Protected` is {surrounded by braces}.  HTML
      and plain text backends just output the text as-is.

    >>> from pybtex.richtext import Protected
    >>> text = Protected('The CTAN archive')
    >>> text.lower()
    Protected('The CTAN archive')
    >>> text.split()
    [Protected('The CTAN archive')]
    >>> print(text.render_as('latex'))
    {The CTAN archive}
    >>> print(text.render_as('html'))
    <span class="bibtex-protected">The CTAN archive</span>

    .. versionadded:: 0.20

    """

    def __init__(self, *args):
        super(Protected, self).__init__(*args)

    def __repr__(self):
        reprparts = ', '.join((repr(part) for part in self.parts))
        return 'Protected({})'.format(reprparts)

    def capfirst(self):
        return self

    def capitalize(self):
        return self

    def lower(self):
        return self

    def upper(self):
        return self

    def split(self, sep=None, keep_empty_parts=None):
        return [self]

    def render(self, backend):
        text = super(Protected, self).render(backend)
        return backend.format_protected(text)