from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
class Recoder:
    """class to return canonical code(s) from code or aliases

    The concept is a lot easier to read in the implementation and
    tests than it is to explain, so...

    >>> # If you have some codes, and several aliases, like this:
    >>> code1 = 1; aliases1=['one', 'first']
    >>> code2 = 2; aliases2=['two', 'second']
    >>> # You might want to do this:
    >>> codes = [[code1]+aliases1,[code2]+aliases2]
    >>> recodes = Recoder(codes)
    >>> recodes.code['one']
    1
    >>> recodes.code['second']
    2
    >>> recodes.code[2]
    2
    >>> # Or maybe you have a code, a label and some aliases
    >>> codes=((1,'label1','one', 'first'),(2,'label2','two'))
    >>> # you might want to get back the code or the label
    >>> recodes = Recoder(codes, fields=('code','label'))
    >>> recodes.code['first']
    1
    >>> recodes.code['label1']
    1
    >>> recodes.label[2]
    'label2'
    >>> # For convenience, you can get the first entered name by
    >>> # indexing the object directly
    >>> recodes[2]
    2
    """
    fields: tuple[str, ...]

    def __init__(self, codes: ty.Sequence[ty.Sequence[ty.Hashable]], fields: ty.Sequence[str]=('code',), map_maker: type[ty.Mapping[ty.Hashable, ty.Hashable]]=dict):
        """Create recoder object

        ``codes`` give a sequence of code, alias sequences
        ``fields`` are names by which the entries in these sequences can be
        accessed.

        By default ``fields`` gives the first column the name
        "code".  The first column is the vector of first entries
        in each of the sequences found in ``codes``.  Thence you can
        get the equivalent first column value with ob.code[value],
        where value can be a first column value, or a value in any of
        the other columns in that sequence.

        You can give other columns names too, and access them in the
        same way - see the examples in the class docstring.

        Parameters
        ----------
        codes : sequence of sequences
            Each sequence defines values (codes) that are equivalent
        fields : {('code',) string sequence}, optional
            names by which elements in sequences can be accessed
        map_maker: callable, optional
            constructor for dict-like objects used to store key value pairs.
            Default is ``dict``.  ``map_maker()`` generates an empty mapping.
            The mapping need only implement ``__getitem__, __setitem__, keys,
            values``.
        """
        self.fields = tuple(fields)
        self.field1 = {}
        for name in fields:
            if name in self.__dict__:
                raise KeyError(f'Input name {name} already in object dict')
            self.__dict__[name] = map_maker()
        self.field1 = self.__dict__[fields[0]]
        self.add_codes(codes)

    def __getattr__(self, key: str) -> ty.Mapping[ty.Hashable, ty.Hashable]:
        raise AttributeError(f'{self.__class__.__name__!r} object has no attribute {key!r}')

    def add_codes(self, code_syn_seqs: ty.Sequence[ty.Sequence[ty.Hashable]]) -> None:
        """Add codes to object

        Parameters
        ----------
        code_syn_seqs : sequence
            sequence of sequences, where each sequence ``S = code_syn_seqs[n]``
            for n in 0..len(code_syn_seqs), is a sequence giving values in the
            same order as ``self.fields``.  Each S should be at least of the
            same length as ``self.fields``.  After this call, if ``self.fields
            == ['field1', 'field2'], then ``self.field1[S[n]] == S[0]`` for all
            n in 0..len(S) and ``self.field2[S[n]] == S[1]`` for all n in
            0..len(S).

        Examples
        --------
        >>> code_syn_seqs = ((2, 'two'), (1, 'one'))
        >>> rc = Recoder(code_syn_seqs)
        >>> rc.value_set() == set((1,2))
        True
        >>> rc.add_codes(((3, 'three'), (1, 'first')))
        >>> rc.value_set() == set((1,2,3))
        True
        >>> print(rc.value_set())  # set is actually ordered
        OrderedSet([2, 1, 3])
        """
        for code_syns in code_syn_seqs:
            for alias in code_syns:
                for field_ind, field_name in enumerate(self.fields):
                    self.__dict__[field_name][alias] = code_syns[field_ind]

    def __getitem__(self, key: ty.Hashable) -> ty.Hashable:
        """Return value from field1 dictionary (first column of values)

        Returns same value as ``obj.field1[key]`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code[key]``

        >>> codes = ((1, 'one'), (2, 'two'))
        >>> Recoder(codes)['two']
        2
        """
        return self.field1[key]

    def __contains__(self, key: ty.Hashable) -> bool:
        """True if field1 in recoder contains `key`"""
        return key in self.field1

    def keys(self):
        """Return all available code and alias values

        Returns same value as ``obj.field1.keys()`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code.keys()``

        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> k = Recoder(codes).keys()
        >>> set(k) == set([1, 2, 'one', 'repeat value', 'two'])
        True
        """
        return self.field1.keys()

    def value_set(self, name: str | None=None) -> OrderedSet:
        """Return OrderedSet of possible returned values for column

        By default, the column is the first column.

        Returns same values as ``set(obj.field1.values())`` and,
        with the default initializing``fields`` argument of
        fields=('code',), this will return the same as
        ``set(obj.code.values())``

        Parameters
        ----------
        name : {None, string}
            Where default of none gives result for first column

        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> vs = Recoder(codes).value_set()
        >>> vs == set([1, 2]) # Sets are not ordered, hence this test
        True
        >>> rc = Recoder(codes, fields=('code', 'label'))
        >>> rc.value_set('label') == set(('one', 'two', 'repeat value'))
        True
        """
        if name is None:
            d = self.field1
        else:
            d = self.__dict__[name]
        return OrderedSet(d.values())