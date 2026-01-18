import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
class BerkeleyDB(Store):
    """    A store that allows for on-disk persistent using BerkeleyDB, a fast
    key/value DB.

    This store implementation used to be known, previous to rdflib 6.0.0
    as 'Sleepycat' due to that being the then name of the Python wrapper
    for BerkeleyDB.

    This store allows for quads as well as triples. See examples of use
    in both the `examples.berkeleydb_example` and ``test/test_store/test_store_berkeleydb.py``
    files.

    **NOTE on installation**:

    To use this store, you must have BerkeleyDB installed on your system
    separately to Python (``brew install berkeley-db`` on a Mac) and also have
    the BerkeleyDB Python wrapper installed (``pip install berkeleydb``).
    You may need to install BerkeleyDB Python wrapper like this:
    ``YES_I_HAVE_THE_RIGHT_TO_USE_THIS_BERKELEY_DB_VERSION=1 pip install berkeleydb``
    """
    context_aware = True
    formula_aware = True
    transaction_aware = False
    graph_aware = True
    db_env: 'db.DBEnv' = None

    def __init__(self, configuration: Optional[str]=None, identifier: Optional['Identifier']=None):
        if not has_bsddb:
            raise ImportError('Unable to import berkeleydb, store is unusable.')
        self.__open = False
        self.__identifier = identifier
        super(BerkeleyDB, self).__init__(configuration)
        self._loads = self.node_pickler.loads
        self._dumps = self.node_pickler.dumps
        self.__indicies_info: List[Tuple[Any, _ToKeyFunc, _FromKeyFunc]]

    def __get_identifier(self) -> Optional['Identifier']:
        return self.__identifier
    identifier = property(__get_identifier)

    def _init_db_environment(self, homeDir: str, create: bool=True) -> 'db.DBEnv':
        if not exists(homeDir):
            if create is True:
                mkdir(homeDir)
                self.create(homeDir)
            else:
                return NO_STORE
        db_env = db.DBEnv()
        db_env.set_cachesize(0, CACHESIZE)
        db_env.set_flags(ENVSETFLAGS, 1)
        db_env.open(homeDir, ENVFLAGS | db.DB_CREATE)
        return db_env

    def is_open(self) -> bool:
        return self.__open

    def open(self, path: str, create: bool=True) -> Optional[int]:
        if not has_bsddb:
            return NO_STORE
        homeDir = path
        if self.__identifier is None:
            self.__identifier = URIRef(pathname2url(abspath(homeDir)))
        db_env = self._init_db_environment(homeDir, create)
        if db_env == NO_STORE:
            return NO_STORE
        self.db_env = db_env
        self.__open = True
        dbname = None
        dbtype = db.DB_BTREE
        dbopenflags = DBOPENFLAGS
        if self.transaction_aware is True:
            dbopenflags |= db.DB_AUTO_COMMIT
        if create:
            dbopenflags |= db.DB_CREATE
        dbmode = 432
        dbsetflags = 0
        self.__indicies: List['db.DB'] = [None] * 3
        self.__indicies_info = [None] * 3
        for i in range(0, 3):
            index_name = to_key_func(i)(('s'.encode('latin-1'), 'p'.encode('latin-1'), 'o'.encode('latin-1')), 'c'.encode('latin-1')).decode()
            index = db.DB(db_env)
            index.set_flags(dbsetflags)
            index.open(index_name, dbname, dbtype, dbopenflags, dbmode)
            self.__indicies[i] = index
            self.__indicies_info[i] = (index, to_key_func(i), from_key_func(i))
        lookup: Dict[int, Tuple['db.DB', _GetPrefixFunc, _FromKeyFunc, _ResultsFromKeyFunc]] = {}
        for i in range(0, 8):
            results: List[Tuple[Tuple[int, int], int, int]] = []
            for start in range(0, 3):
                score = 1
                len = 0
                for j in range(start, start + 3):
                    if i & 1 << j % 3:
                        score = score << 1
                        len += 1
                    else:
                        break
                tie_break = 2 - start
                results.append(((score, tie_break), start, len))
            results.sort()
            score, start, len = results[-1]

            def get_prefix_func(start: int, end: int) -> _GetPrefixFunc:

                def get_prefix(triple: Tuple[str, str, str], context: Optional[str]) -> Generator[str, None, None]:
                    if context is None:
                        yield ''
                    else:
                        yield context
                    i = start
                    while i < end:
                        yield triple[i % 3]
                        i += 1
                    yield ''
                return get_prefix
            lookup[i] = (self.__indicies[start], get_prefix_func(start, start + len), from_key_func(start), results_from_key_func(start, self._from_string))
        self.__lookup_dict = lookup
        self.__contexts = db.DB(db_env)
        self.__contexts.set_flags(dbsetflags)
        self.__contexts.open('contexts', dbname, dbtype, dbopenflags, dbmode)
        self.__namespace = db.DB(db_env)
        self.__namespace.set_flags(dbsetflags)
        self.__namespace.open('namespace', dbname, dbtype, dbopenflags, dbmode)
        self.__prefix = db.DB(db_env)
        self.__prefix.set_flags(dbsetflags)
        self.__prefix.open('prefix', dbname, dbtype, dbopenflags, dbmode)
        self.__k2i = db.DB(db_env)
        self.__k2i.set_flags(dbsetflags)
        self.__k2i.open('k2i', dbname, db.DB_HASH, dbopenflags, dbmode)
        self.__i2k = db.DB(db_env)
        self.__i2k.set_flags(dbsetflags)
        self.__i2k.open('i2k', dbname, db.DB_RECNO, dbopenflags, dbmode)
        self.__needs_sync = False
        t = Thread(target=self.__sync_run)
        t.setDaemon(True)
        t.start()
        self.__sync_thread = t
        return VALID_STORE

    def __sync_run(self) -> None:
        from time import sleep, time
        try:
            min_seconds, max_seconds = (10, 300)
            while self.__open:
                if self.__needs_sync:
                    t0 = t1 = time()
                    self.__needs_sync = False
                    while self.__open:
                        sleep(0.1)
                        if self.__needs_sync:
                            t1 = time()
                            self.__needs_sync = False
                        if time() - t1 > min_seconds or time() - t0 > max_seconds:
                            self.__needs_sync = False
                            logger.debug('sync')
                            self.sync()
                            break
                else:
                    sleep(1)
        except Exception as e:
            logger.exception(e)

    def sync(self) -> None:
        if self.__open:
            for i in self.__indicies:
                i.sync()
            self.__contexts.sync()
            self.__namespace.sync()
            self.__prefix.sync()
            self.__i2k.sync()
            self.__k2i.sync()

    def close(self, commit_pending_transaction: bool=False) -> None:
        self.__open = False
        self.__sync_thread.join()
        for i in self.__indicies:
            i.close()
        self.__contexts.close()
        self.__namespace.close()
        self.__prefix.close()
        self.__i2k.close()
        self.__k2i.close()
        self.db_env.close()

    def add(self, triple: '_TripleType', context: '_ContextType', quoted: bool=False, txn: Optional[Any]=None) -> None:
        """        Add a triple to the store of triples.
        """
        subject, predicate, object = triple
        assert self.__open, 'The Store must be open.'
        assert context != self, 'Can not add triple directly to store'
        Store.add(self, (subject, predicate, object), context, quoted)
        _to_string = self._to_string
        s = _to_string(subject, txn=txn)
        p = _to_string(predicate, txn=txn)
        o = _to_string(object, txn=txn)
        c = _to_string(context, txn=txn)
        cspo, cpos, cosp = self.__indicies
        value = cspo.get(bb('%s^%s^%s^%s^' % (c, s, p, o)), txn=txn)
        if value is None:
            self.__contexts.put(bb(c), b'', txn=txn)
            contexts_value = cspo.get(bb('%s^%s^%s^%s^' % ('', s, p, o)), txn=txn) or ''.encode('latin-1')
            contexts = set(contexts_value.split('^'.encode('latin-1')))
            contexts.add(bb(c))
            contexts_value = '^'.encode('latin-1').join(contexts)
            assert contexts_value is not None
            cspo.put(bb('%s^%s^%s^%s^' % (c, s, p, o)), b'', txn=txn)
            cpos.put(bb('%s^%s^%s^%s^' % (c, p, o, s)), b'', txn=txn)
            cosp.put(bb('%s^%s^%s^%s^' % (c, o, s, p)), b'', txn=txn)
            if not quoted:
                cspo.put(bb('%s^%s^%s^%s^' % ('', s, p, o)), contexts_value, txn=txn)
                cpos.put(bb('%s^%s^%s^%s^' % ('', p, o, s)), contexts_value, txn=txn)
                cosp.put(bb('%s^%s^%s^%s^' % ('', o, s, p)), contexts_value, txn=txn)
            self.__needs_sync = True

    def __remove(self, spo: Tuple[bytes, bytes, bytes], c: bytes, quoted: bool=False, txn: Optional[Any]=None) -> None:
        s, p, o = spo
        cspo, cpos, cosp = self.__indicies
        contexts_value = cspo.get('^'.encode('latin-1').join([''.encode('latin-1'), s, p, o, ''.encode('latin-1')]), txn=txn) or ''.encode('latin-1')
        contexts = set(contexts_value.split('^'.encode('latin-1')))
        contexts.discard(c)
        contexts_value = '^'.encode('latin-1').join(contexts)
        for i, _to_key, _from_key in self.__indicies_info:
            i.delete(_to_key((s, p, o), c), txn=txn)
        if not quoted:
            if contexts_value:
                for i, _to_key, _from_key in self.__indicies_info:
                    i.put(_to_key((s, p, o), ''.encode('latin-1')), contexts_value, txn=txn)
            else:
                for i, _to_key, _from_key in self.__indicies_info:
                    try:
                        i.delete(_to_key((s, p, o), ''.encode('latin-1')), txn=txn)
                    except db.DBNotFoundError:
                        pass

    def remove(self, spo: '_TriplePatternType', context: Optional['_ContextType'], txn: Optional[Any]=None) -> None:
        subject, predicate, object = spo
        assert self.__open, 'The Store must be open.'
        Store.remove(self, (subject, predicate, object), context)
        _to_string = self._to_string
        if context is not None:
            if context == self:
                context = None
        if subject is not None and predicate is not None and (object is not None) and (context is not None):
            s = _to_string(subject, txn=txn)
            p = _to_string(predicate, txn=txn)
            o = _to_string(object, txn=txn)
            c = _to_string(context, txn=txn)
            value = self.__indicies[0].get(bb('%s^%s^%s^%s^' % (c, s, p, o)), txn=txn)
            if value is not None:
                self.__remove((bb(s), bb(p), bb(o)), bb(c), txn=txn)
                self.__needs_sync = True
        else:
            cspo, cpos, cosp = self.__indicies
            index, prefix, from_key, results_from_key = self.__lookup((subject, predicate, object), context, txn=txn)
            cursor = index.cursor(txn=txn)
            try:
                current = cursor.set_range(prefix)
                needs_sync = True
            except db.DBNotFoundError:
                current = None
                needs_sync = False
            cursor.close()
            while current:
                key, value = current
                cursor = index.cursor(txn=txn)
                try:
                    cursor.set_range(key)
                    current = getattr(cursor, 'next')()
                except db.DBNotFoundError:
                    current = None
                cursor.close()
                if key.startswith(prefix):
                    c, s, p, o = from_key(key)
                    if context is None:
                        contexts_value = index.get(key, txn=txn) or ''.encode('latin-1')
                        contexts = set(contexts_value.split('^'.encode('latin-1')))
                        contexts.add(''.encode('latin-1'))
                        for c in contexts:
                            for i, _to_key, _ in self.__indicies_info:
                                i.delete(_to_key((s, p, o), c), txn=txn)
                    else:
                        self.__remove((s, p, o), c, txn=txn)
                else:
                    break
            if context is not None:
                if subject is None and predicate is None and (object is None):
                    try:
                        self.__contexts.delete(bb(_to_string(context, txn=txn)), txn=txn)
                    except db.DBNotFoundError:
                        pass
            self.__needs_sync = needs_sync

    def triples(self, spo: '_TriplePatternType', context: Optional['_ContextType']=None, txn: Optional[Any]=None) -> Generator[Tuple['_TripleType', Generator[Optional['_ContextType'], None, None]], None, None]:
        """A generator over all the triples matching"""
        assert self.__open, 'The Store must be open.'
        subject, predicate, object = spo
        if context is not None:
            if context == self:
                context = None
        index, prefix, from_key, results_from_key = self.__lookup((subject, predicate, object), context, txn=txn)
        cursor = index.cursor(txn=txn)
        try:
            current = cursor.set_range(prefix)
        except db.DBNotFoundError:
            current = None
        cursor.close()
        while current:
            key, value = current
            cursor = index.cursor(txn=txn)
            try:
                cursor.set_range(key)
                current = getattr(cursor, 'next')()
            except db.DBNotFoundError:
                current = None
            cursor.close()
            if key and key.startswith(prefix):
                contexts_value = index.get(key, txn=txn)
                yield results_from_key(key, subject, predicate, object, contexts_value)
            else:
                break

    def __len__(self, context: Optional['_ContextType']=None) -> int:
        assert self.__open, 'The Store must be open.'
        if context is not None:
            if context == self:
                context = None
        if context is None:
            prefix = '^'.encode('latin-1')
        else:
            prefix = bb('%s^' % self._to_string(context))
        index = self.__indicies[0]
        cursor = index.cursor()
        current = cursor.set_range(prefix)
        count = 0
        while current:
            key, value = current
            if key.startswith(prefix):
                count += 1
                current = getattr(cursor, 'next')()
            else:
                break
        cursor.close()
        return count

    def bind(self, prefix: str, namespace: 'URIRef', override: bool=True) -> None:
        prefix = prefix.encode('utf-8')
        namespace = namespace.encode('utf-8')
        bound_prefix = self.__prefix.get(namespace)
        bound_namespace = self.__namespace.get(prefix)
        if override:
            if bound_prefix:
                self.__namespace.delete(bound_prefix)
            if bound_namespace:
                self.__prefix.delete(bound_namespace)
            self.__prefix[namespace] = prefix
            self.__namespace[prefix] = namespace
        else:
            self.__prefix[bound_namespace or namespace] = bound_prefix or prefix
            self.__namespace[bound_prefix or prefix] = bound_namespace or namespace

    def namespace(self, prefix: str) -> Optional['URIRef']:
        prefix = prefix.encode('utf-8')
        ns = self.__namespace.get(prefix, None)
        if ns is not None:
            return URIRef(ns.decode('utf-8'))
        return None

    def prefix(self, namespace: 'URIRef') -> Optional[str]:
        namespace = namespace.encode('utf-8')
        prefix = self.__prefix.get(namespace, None)
        if prefix is not None:
            return prefix.decode('utf-8')
        return None

    def namespaces(self) -> Generator[Tuple[str, 'URIRef'], None, None]:
        cursor = self.__namespace.cursor()
        results = []
        current = cursor.first()
        while current:
            prefix, namespace = current
            results.append((prefix.decode('utf-8'), namespace.decode('utf-8')))
            current = getattr(cursor, 'next')()
        cursor.close()
        for prefix, namespace in results:
            yield (prefix, URIRef(namespace))

    def contexts(self, triple: Optional['_TripleType']=None) -> Generator['_ContextType', None, None]:
        _from_string = self._from_string
        _to_string = self._to_string
        if triple:
            s: str
            p: str
            o: str
            s, p, o = triple
            s = _to_string(s)
            p = _to_string(p)
            o = _to_string(o)
            contexts = self.__indicies[0].get(bb('%s^%s^%s^%s^' % ('', s, p, o)))
            if contexts:
                for c in contexts.split('^'.encode('latin-1')):
                    if c:
                        yield _from_string(c)
        else:
            index = self.__contexts
            cursor = index.cursor()
            current = cursor.first()
            cursor.close()
            while current:
                key, value = current
                context = _from_string(key)
                yield context
                cursor = index.cursor()
                try:
                    cursor.set_range(key)
                    current = getattr(cursor, 'next')()
                except db.DBNotFoundError:
                    current = None
                cursor.close()

    def add_graph(self, graph: 'Graph') -> None:
        self.__contexts.put(bb(self._to_string(graph)), b'')

    def remove_graph(self, graph: 'Graph'):
        self.remove((None, None, None), graph)

    def _from_string(self, i: bytes) -> Node:
        k = self.__i2k.get(int(i))
        return self._loads(k)

    def _to_string(self, term: Node, txn: Optional[Any]=None) -> str:
        k = self._dumps(term)
        i = self.__k2i.get(k, txn=txn)
        if i is None:
            if self.transaction_aware:
                i = '%s' % self.__i2k.append(k, txn)
            else:
                i = '%s' % self.__i2k.append(k)
            self.__k2i.put(k, i.encode(), txn=txn)
        else:
            i = i.decode()
        return i

    def __lookup(self, spo: '_TriplePatternType', context: Optional['_ContextType'], txn: Optional[Any]=None) -> Tuple['db.DB', bytes, _FromKeyFunc, _ResultsFromKeyFunc]:
        subject, predicate, object = spo
        _to_string = self._to_string
        if context is not None:
            context = _to_string(context, txn=txn)
        i = 0
        if subject is not None:
            i += 1
            subject = _to_string(subject, txn=txn)
        if predicate is not None:
            i += 2
            predicate = _to_string(predicate, txn=txn)
        if object is not None:
            i += 4
            object = _to_string(object, txn=txn)
        index, prefix_func, from_key, results_from_key = self.__lookup_dict[i]
        prefix = bb('^'.join(prefix_func((subject, predicate, object), context)))
        return (index, prefix, from_key, results_from_key)