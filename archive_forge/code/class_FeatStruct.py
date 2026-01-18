import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
@total_ordering
class FeatStruct(SubstituteBindingsI):
    """
    A mapping from feature identifiers to feature values, where each
    feature value is either a basic value (such as a string or an
    integer), or a nested feature structure.  There are two types of
    feature structure:

      - feature dictionaries, implemented by ``FeatDict``, act like
        Python dictionaries.  Feature identifiers may be strings or
        instances of the ``Feature`` class.
      - feature lists, implemented by ``FeatList``, act like Python
        lists.  Feature identifiers are integers.

    Feature structures may be indexed using either simple feature
    identifiers or 'feature paths.'  A feature path is a sequence
    of feature identifiers that stand for a corresponding sequence of
    indexing operations.  In particular, ``fstruct[(f1,f2,...,fn)]`` is
    equivalent to ``fstruct[f1][f2]...[fn]``.

    Feature structures may contain reentrant feature structures.  A
    "reentrant feature structure" is a single feature structure
    object that can be accessed via multiple feature paths.  Feature
    structures may also be cyclic.  A feature structure is "cyclic"
    if there is any feature path from the feature structure to itself.

    Two feature structures are considered equal if they assign the
    same values to all features, and have the same reentrancies.

    By default, feature structures are mutable.  They may be made
    immutable with the ``freeze()`` method.  Once they have been
    frozen, they may be hashed, and thus used as dictionary keys.
    """
    _frozen = False
    ':ivar: A flag indicating whether this feature structure is\n       frozen or not.  Once this flag is set, it should never be\n       un-set; and no further modification should be made to this\n       feature structure.'

    def __new__(cls, features=None, **morefeatures):
        """
        Construct and return a new feature structure.  If this
        constructor is called directly, then the returned feature
        structure will be an instance of either the ``FeatDict`` class
        or the ``FeatList`` class.

        :param features: The initial feature values for this feature
            structure:

            - FeatStruct(string) -> FeatStructReader().read(string)
            - FeatStruct(mapping) -> FeatDict(mapping)
            - FeatStruct(sequence) -> FeatList(sequence)
            - FeatStruct() -> FeatDict()
        :param morefeatures: If ``features`` is a mapping or None,
            then ``morefeatures`` provides additional features for the
            ``FeatDict`` constructor.
        """
        if cls is FeatStruct:
            if features is None:
                return FeatDict.__new__(FeatDict, **morefeatures)
            elif _is_mapping(features):
                return FeatDict.__new__(FeatDict, features, **morefeatures)
            elif morefeatures:
                raise TypeError('Keyword arguments may only be specified if features is None or is a mapping.')
            if isinstance(features, str):
                if FeatStructReader._START_FDICT_RE.match(features):
                    return FeatDict.__new__(FeatDict, features, **morefeatures)
                else:
                    return FeatList.__new__(FeatList, features, **morefeatures)
            elif _is_sequence(features):
                return FeatList.__new__(FeatList, features)
            else:
                raise TypeError('Expected string or mapping or sequence')
        else:
            return super().__new__(cls, features, **morefeatures)

    def _keys(self):
        """Return an iterable of the feature identifiers used by this
        FeatStruct."""
        raise NotImplementedError()

    def _values(self):
        """Return an iterable of the feature values directly defined
        by this FeatStruct."""
        raise NotImplementedError()

    def _items(self):
        """Return an iterable of (fid,fval) pairs, where fid is a
        feature identifier and fval is the corresponding feature
        value, for all features defined by this FeatStruct."""
        raise NotImplementedError()

    def equal_values(self, other, check_reentrance=False):
        """
        Return True if ``self`` and ``other`` assign the same value to
        to every feature.  In particular, return true if
        ``self[p]==other[p]`` for every feature path *p* such
        that ``self[p]`` or ``other[p]`` is a base value (i.e.,
        not a nested feature structure).

        :param check_reentrance: If True, then also return False if
            there is any difference between the reentrances of ``self``
            and ``other``.
        :note: the ``==`` is equivalent to ``equal_values()`` with
            ``check_reentrance=True``.
        """
        return self._equal(other, check_reentrance, set(), set(), set())

    def __eq__(self, other):
        """
        Return true if ``self`` and ``other`` are both feature structures,
        assign the same values to all features, and contain the same
        reentrances.  I.e., return
        ``self.equal_values(other, check_reentrance=True)``.

        :see: ``equal_values()``
        """
        return self._equal(other, True, set(), set(), set())

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, FeatStruct):
            return self.__class__.__name__ < other.__class__.__name__
        else:
            return len(self) < len(other)

    def __hash__(self):
        """
        If this feature structure is frozen, return its hash value;
        otherwise, raise ``TypeError``.
        """
        if not self._frozen:
            raise TypeError('FeatStructs must be frozen before they can be hashed.')
        try:
            return self._hash
        except AttributeError:
            self._hash = self._calculate_hashvalue(set())
            return self._hash

    def _equal(self, other, check_reentrance, visited_self, visited_other, visited_pairs):
        """
        Return True iff self and other have equal values.

        :param visited_self: A set containing the ids of all ``self``
            feature structures we've already visited.
        :param visited_other: A set containing the ids of all ``other``
            feature structures we've already visited.
        :param visited_pairs: A set containing ``(selfid, otherid)`` pairs
            for all pairs of feature structures we've already visited.
        """
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if len(self) != len(other):
            return False
        if set(self._keys()) != set(other._keys()):
            return False
        if check_reentrance:
            if id(self) in visited_self or id(other) in visited_other:
                return (id(self), id(other)) in visited_pairs
        elif (id(self), id(other)) in visited_pairs:
            return True
        visited_self.add(id(self))
        visited_other.add(id(other))
        visited_pairs.add((id(self), id(other)))
        for fname, self_fval in self._items():
            other_fval = other[fname]
            if isinstance(self_fval, FeatStruct):
                if not self_fval._equal(other_fval, check_reentrance, visited_self, visited_other, visited_pairs):
                    return False
            elif self_fval != other_fval:
                return False
        return True

    def _calculate_hashvalue(self, visited):
        """
        Return a hash value for this feature structure.

        :require: ``self`` must be frozen.
        :param visited: A set containing the ids of all feature
            structures we've already visited while hashing.
        """
        if id(self) in visited:
            return 1
        visited.add(id(self))
        hashval = 5831
        for fname, fval in sorted(self._items()):
            hashval *= 37
            hashval += hash(fname)
            hashval *= 37
            if isinstance(fval, FeatStruct):
                hashval += fval._calculate_hashvalue(visited)
            else:
                hashval += hash(fval)
            hashval = int(hashval & 2147483647)
        return hashval
    _FROZEN_ERROR = 'Frozen FeatStructs may not be modified.'

    def freeze(self):
        """
        Make this feature structure, and any feature structures it
        contains, immutable.  Note: this method does not attempt to
        'freeze' any feature value that is not a ``FeatStruct``; it
        is recommended that you use only immutable feature values.
        """
        if self._frozen:
            return
        self._freeze(set())

    def frozen(self):
        """
        Return True if this feature structure is immutable.  Feature
        structures can be made immutable with the ``freeze()`` method.
        Immutable feature structures may not be made mutable again,
        but new mutable copies can be produced with the ``copy()`` method.
        """
        return self._frozen

    def _freeze(self, visited):
        """
        Make this feature structure, and any feature structure it
        contains, immutable.

        :param visited: A set containing the ids of all feature
            structures we've already visited while freezing.
        """
        if id(self) in visited:
            return
        visited.add(id(self))
        self._frozen = True
        for fname, fval in sorted(self._items()):
            if isinstance(fval, FeatStruct):
                fval._freeze(visited)

    def copy(self, deep=True):
        """
        Return a new copy of ``self``.  The new copy will not be frozen.

        :param deep: If true, create a deep copy; if false, create
            a shallow copy.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return self.__class__(self)

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def cyclic(self):
        """
        Return True if this feature structure contains itself.
        """
        return self._find_reentrances({})[id(self)]

    def walk(self):
        """
        Return an iterator that generates this feature structure, and
        each feature structure it contains.  Each feature structure will
        be generated exactly once.
        """
        return self._walk(set())

    def _walk(self, visited):
        """
        Return an iterator that generates this feature structure, and
        each feature structure it contains.

        :param visited: A set containing the ids of all feature
            structures we've already visited while freezing.
        """
        raise NotImplementedError()

    def _walk(self, visited):
        if id(self) in visited:
            return
        visited.add(id(self))
        yield self
        for fval in self._values():
            if isinstance(fval, FeatStruct):
                yield from fval._walk(visited)

    def _find_reentrances(self, reentrances):
        """
        Return a dictionary that maps from the ``id`` of each feature
        structure contained in ``self`` (including ``self``) to a
        boolean value, indicating whether it is reentrant or not.
        """
        if id(self) in reentrances:
            reentrances[id(self)] = True
        else:
            reentrances[id(self)] = False
            for fval in self._values():
                if isinstance(fval, FeatStruct):
                    fval._find_reentrances(reentrances)
        return reentrances

    def substitute_bindings(self, bindings):
        """:see: ``nltk.featstruct.substitute_bindings()``"""
        return substitute_bindings(self, bindings)

    def retract_bindings(self, bindings):
        """:see: ``nltk.featstruct.retract_bindings()``"""
        return retract_bindings(self, bindings)

    def variables(self):
        """:see: ``nltk.featstruct.find_variables()``"""
        return find_variables(self)

    def rename_variables(self, vars=None, used_vars=(), new_vars=None):
        """:see: ``nltk.featstruct.rename_variables()``"""
        return rename_variables(self, vars, used_vars, new_vars)

    def remove_variables(self):
        """
        Return the feature structure that is obtained by deleting
        any feature whose value is a ``Variable``.

        :rtype: FeatStruct
        """
        return remove_variables(self)

    def unify(self, other, bindings=None, trace=False, fail=None, rename_vars=True):
        return unify(self, other, bindings, trace, fail, rename_vars)

    def subsumes(self, other):
        """
        Return True if ``self`` subsumes ``other``.  I.e., return true
        If unifying ``self`` with ``other`` would result in a feature
        structure equal to ``other``.
        """
        return subsumes(self, other)

    def __repr__(self):
        """
        Display a single-line representation of this feature structure,
        suitable for embedding in other representations.
        """
        return self._repr(self._find_reentrances({}), {})

    def _repr(self, reentrances, reentrance_ids):
        """
        Return a string representation of this feature structure.

        :param reentrances: A dictionary that maps from the ``id`` of
            each feature value in self, indicating whether that value
            is reentrant or not.
        :param reentrance_ids: A dictionary mapping from each ``id``
            of a feature value to a unique identifier.  This is modified
            by ``repr``: the first time a reentrant feature value is
            displayed, an identifier is added to ``reentrance_ids`` for it.
        """
        raise NotImplementedError()