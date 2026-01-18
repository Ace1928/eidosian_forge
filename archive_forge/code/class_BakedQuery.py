import collections.abc as collections_abc
import logging
from .. import exc as sa_exc
from .. import util
from ..orm import exc as orm_exc
from ..orm.query import Query
from ..orm.session import Session
from ..sql import func
from ..sql import literal_column
from ..sql import util as sql_util
class BakedQuery:
    """A builder object for :class:`.query.Query` objects."""
    __slots__ = ('steps', '_bakery', '_cache_key', '_spoiled')

    def __init__(self, bakery, initial_fn, args=()):
        self._cache_key = ()
        self._update_cache_key(initial_fn, args)
        self.steps = [initial_fn]
        self._spoiled = False
        self._bakery = bakery

    @classmethod
    def bakery(cls, size=200, _size_alert=None):
        """Construct a new bakery.

        :return: an instance of :class:`.Bakery`

        """
        return Bakery(cls, util.LRUCache(size, size_alert=_size_alert))

    def _clone(self):
        b1 = BakedQuery.__new__(BakedQuery)
        b1._cache_key = self._cache_key
        b1.steps = list(self.steps)
        b1._bakery = self._bakery
        b1._spoiled = self._spoiled
        return b1

    def _update_cache_key(self, fn, args=()):
        self._cache_key += (fn.__code__,) + args

    def __iadd__(self, other):
        if isinstance(other, tuple):
            self.add_criteria(*other)
        else:
            self.add_criteria(other)
        return self

    def __add__(self, other):
        if isinstance(other, tuple):
            return self.with_criteria(*other)
        else:
            return self.with_criteria(other)

    def add_criteria(self, fn, *args):
        """Add a criteria function to this :class:`.BakedQuery`.

        This is equivalent to using the ``+=`` operator to
        modify a :class:`.BakedQuery` in-place.

        """
        self._update_cache_key(fn, args)
        self.steps.append(fn)
        return self

    def with_criteria(self, fn, *args):
        """Add a criteria function to a :class:`.BakedQuery` cloned from this
        one.

        This is equivalent to using the ``+`` operator to
        produce a new :class:`.BakedQuery` with modifications.

        """
        return self._clone().add_criteria(fn, *args)

    def for_session(self, session):
        """Return a :class:`_baked.Result` object for this
        :class:`.BakedQuery`.

        This is equivalent to calling the :class:`.BakedQuery` as a
        Python callable, e.g. ``result = my_baked_query(session)``.

        """
        return Result(self, session)

    def __call__(self, session):
        return self.for_session(session)

    def spoil(self, full=False):
        """Cancel any query caching that will occur on this BakedQuery object.

        The BakedQuery can continue to be used normally, however additional
        creational functions will not be cached; they will be called
        on every invocation.

        This is to support the case where a particular step in constructing
        a baked query disqualifies the query from being cacheable, such
        as a variant that relies upon some uncacheable value.

        :param full: if False, only functions added to this
         :class:`.BakedQuery` object subsequent to the spoil step will be
         non-cached; the state of the :class:`.BakedQuery` up until
         this point will be pulled from the cache.   If True, then the
         entire :class:`_query.Query` object is built from scratch each
         time, with all creational functions being called on each
         invocation.

        """
        if not full and (not self._spoiled):
            _spoil_point = self._clone()
            _spoil_point._cache_key += ('_query_only',)
            self.steps = [_spoil_point._retrieve_baked_query]
        self._spoiled = True
        return self

    def _effective_key(self, session):
        """Return the key that actually goes into the cache dictionary for
        this :class:`.BakedQuery`, taking into account the given
        :class:`.Session`.

        This basically means we also will include the session's query_class,
        as the actual :class:`_query.Query` object is part of what's cached
        and needs to match the type of :class:`_query.Query` that a later
        session will want to use.

        """
        return self._cache_key + (session._query_cls,)

    def _with_lazyload_options(self, options, effective_path, cache_path=None):
        """Cloning version of _add_lazyload_options."""
        q = self._clone()
        q._add_lazyload_options(options, effective_path, cache_path=cache_path)
        return q

    def _add_lazyload_options(self, options, effective_path, cache_path=None):
        """Used by per-state lazy loaders to add options to the
        "lazy load" query from a parent query.

        Creates a cache key based on given load path and query options;
        if a repeatable cache key cannot be generated, the query is
        "spoiled" so that it won't use caching.

        """
        key = ()
        if not cache_path:
            cache_path = effective_path
        for opt in options:
            if opt._is_legacy_option or opt._is_compile_state:
                ck = opt._generate_cache_key()
                if ck is None:
                    self.spoil(full=True)
                else:
                    assert not ck[1], 'loader options with variable bound parameters not supported with baked queries.  Please use new-style select() statements for cached ORM queries.'
                    key += ck[0]
        self.add_criteria(lambda q: q._with_current_path(effective_path).options(*options), cache_path.path, key)

    def _retrieve_baked_query(self, session):
        query = self._bakery.get(self._effective_key(session), None)
        if query is None:
            query = self._as_query(session)
            self._bakery[self._effective_key(session)] = query.with_session(None)
        return query.with_session(session)

    def _bake(self, session):
        query = self._as_query(session)
        query.session = None
        statement = query._statement_20()
        if statement._compile_options._bake_ok:
            self._bakery[self._effective_key(session)] = (query, statement)
        return (query, statement)

    def to_query(self, query_or_session):
        """Return the :class:`_query.Query` object for use as a subquery.

        This method should be used within the lambda callable being used
        to generate a step of an enclosing :class:`.BakedQuery`.   The
        parameter should normally be the :class:`_query.Query` object that
        is passed to the lambda::

            sub_bq = self.bakery(lambda s: s.query(User.name))
            sub_bq += lambda q: q.filter(
                User.id == Address.user_id).correlate(Address)

            main_bq = self.bakery(lambda s: s.query(Address))
            main_bq += lambda q: q.filter(
                sub_bq.to_query(q).exists())

        In the case where the subquery is used in the first callable against
        a :class:`.Session`, the :class:`.Session` is also accepted::

            sub_bq = self.bakery(lambda s: s.query(User.name))
            sub_bq += lambda q: q.filter(
                User.id == Address.user_id).correlate(Address)

            main_bq = self.bakery(
                lambda s: s.query(
                Address.id, sub_bq.to_query(q).scalar_subquery())
            )

        :param query_or_session: a :class:`_query.Query` object or a class
         :class:`.Session` object, that is assumed to be within the context
         of an enclosing :class:`.BakedQuery` callable.


         .. versionadded:: 1.3


        """
        if isinstance(query_or_session, Session):
            session = query_or_session
        elif isinstance(query_or_session, Query):
            session = query_or_session.session
            if session is None:
                raise sa_exc.ArgumentError('Given Query needs to be associated with a Session')
        else:
            raise TypeError('Query or Session object expected, got %r.' % type(query_or_session))
        return self._as_query(session)

    def _as_query(self, session):
        query = self.steps[0](session)
        for step in self.steps[1:]:
            query = step(query)
        return query