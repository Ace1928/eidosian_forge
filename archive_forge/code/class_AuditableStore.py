import threading
from typing import TYPE_CHECKING, Any, Generator, Iterator, List, Optional, Tuple
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.store import Store
class AuditableStore(Store):

    def __init__(self, store: 'Store'):
        self.store = store
        self.context_aware = store.context_aware
        self.formula_aware = False
        self.transaction_aware = True
        self.reverseOps: List[Tuple[Optional['_SubjectType'], Optional['_PredicateType'], Optional['_ObjectType'], Optional['_ContextIdentifierType'], str]] = []
        self.rollbackLock = threading.RLock()

    def open(self, configuration: str, create: bool=True) -> Optional[int]:
        return self.store.open(configuration, create)

    def close(self, commit_pending_transaction: bool=False) -> None:
        self.store.close()

    def destroy(self, configuration: str) -> None:
        self.store.destroy(configuration)

    def query(self, *args: Any, **kw: Any) -> 'Result':
        return self.store.query(*args, **kw)

    def add(self, triple: '_TripleType', context: '_ContextType', quoted: bool=False) -> None:
        s, p, o = triple
        lock = destructiveOpLocks['add']
        lock = lock if lock else threading.RLock()
        with lock:
            context = context.__class__(self.store, context.identifier) if context is not None else None
            ctxId = context.identifier if context is not None else None
            if list(self.store.triples(triple, context)):
                return
            self.reverseOps.append((s, p, o, ctxId, 'remove'))
            try:
                self.reverseOps.remove((s, p, o, ctxId, 'add'))
            except ValueError:
                pass
            self.store.add((s, p, o), context, quoted)

    def remove(self, spo: '_TriplePatternType', context: Optional['_ContextType']=None) -> None:
        subject, predicate, object_ = spo
        lock = destructiveOpLocks['remove']
        lock = lock if lock else threading.RLock()
        with lock:
            context = context.__class__(self.store, context.identifier) if context is not None else None
            ctxId = context.identifier if context is not None else None
            if None in [subject, predicate, object_, context]:
                if ctxId:
                    for s, p, o in context.triples((subject, predicate, object_)):
                        try:
                            self.reverseOps.remove((s, p, o, ctxId, 'remove'))
                        except ValueError:
                            self.reverseOps.append((s, p, o, ctxId, 'add'))
                else:
                    for s, p, o, ctx in ConjunctiveGraph(self.store).quads((subject, predicate, object_)):
                        try:
                            self.reverseOps.remove((s, p, o, ctx.identifier, 'remove'))
                        except ValueError:
                            self.reverseOps.append((s, p, o, ctx.identifier, 'add'))
            else:
                if not list(self.triples((subject, predicate, object_), context)):
                    return
                try:
                    self.reverseOps.remove((subject, predicate, object_, ctxId, 'remove'))
                except ValueError:
                    self.reverseOps.append((subject, predicate, object_, ctxId, 'add'))
            self.store.remove((subject, predicate, object_), context)

    def triples(self, triple: '_TriplePatternType', context: Optional['_ContextType']=None) -> Iterator[Tuple['_TripleType', Iterator[Optional['_ContextType']]]]:
        su, pr, ob = triple
        context = context.__class__(self.store, context.identifier) if context is not None else None
        for (s, p, o), cg in self.store.triples((su, pr, ob), context):
            yield ((s, p, o), cg)

    def __len__(self, context: Optional['_ContextType']=None):
        context = context.__class__(self.store, context.identifier) if context is not None else None
        return self.store.__len__(context)

    def contexts(self, triple: Optional['_TripleType']=None) -> Generator['_ContextType', None, None]:
        for ctx in self.store.contexts(triple):
            yield ctx

    def bind(self, prefix: str, namespace: 'URIRef', override: bool=True) -> None:
        self.store.bind(prefix, namespace, override=override)

    def prefix(self, namespace: 'URIRef') -> Optional[str]:
        return self.store.prefix(namespace)

    def namespace(self, prefix: str) -> Optional['URIRef']:
        return self.store.namespace(prefix)

    def namespaces(self) -> Iterator[Tuple[str, 'URIRef']]:
        return self.store.namespaces()

    def commit(self) -> None:
        self.reverseOps = []

    def rollback(self) -> None:
        with self.rollbackLock:
            for subject, predicate, obj, context, op in self.reverseOps:
                if op == 'add':
                    self.store.add((subject, predicate, obj), Graph(self.store, context))
                else:
                    self.store.remove((subject, predicate, obj), Graph(self.store, context))
            self.reverseOps = []