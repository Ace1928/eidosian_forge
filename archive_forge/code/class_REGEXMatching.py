import re
from rdflib.graph import Graph
from rdflib.store import Store
class REGEXMatching(Store):

    def __init__(self, storage):
        self.storage = storage
        self.context_aware = storage.context_aware
        self.formula_aware = storage.formula_aware
        self.transaction_aware = storage.transaction_aware

    def open(self, configuration, create=True):
        return self.storage.open(configuration, create)

    def close(self, commit_pending_transaction=False):
        self.storage.close()

    def destroy(self, configuration):
        self.storage.destroy(configuration)

    def add(self, triple, context, quoted=False):
        subject, predicate, object_ = triple
        self.storage.add((subject, predicate, object_), context, quoted)

    def remove(self, triple, context=None):
        subject, predicate, object_ = triple
        if isinstance(subject, REGEXTerm) or isinstance(predicate, REGEXTerm) or isinstance(object_, REGEXTerm) or (context is not None and isinstance(context.identifier, REGEXTerm)):
            s = not isinstance(subject, REGEXTerm) and subject or None
            p = not isinstance(predicate, REGEXTerm) and predicate or None
            o = not isinstance(object_, REGEXTerm) and object_ or None
            c = (context is not None and (not isinstance(context.identifier, REGEXTerm))) and context or None
            removeQuadList = []
            for (s1, p1, o1), cg in self.storage.triples((s, p, o), c):
                for ctx in cg:
                    ctx = ctx.identifier
                    if regexCompareQuad((s1, p1, o1, ctx), (subject, predicate, object_, context is not None and context.identifier or context)):
                        removeQuadList.append((s1, p1, o1, ctx))
            for s, p, o, c in removeQuadList:
                self.storage.remove((s, p, o), c and Graph(self, c) or c)
        else:
            self.storage.remove((subject, predicate, object_), context)

    def triples(self, triple, context=None):
        subject, predicate, object_ = triple
        if isinstance(subject, REGEXTerm) or isinstance(predicate, REGEXTerm) or isinstance(object_, REGEXTerm) or (context is not None and isinstance(context.identifier, REGEXTerm)):
            s = not isinstance(subject, REGEXTerm) and subject or None
            p = not isinstance(predicate, REGEXTerm) and predicate or None
            o = not isinstance(object_, REGEXTerm) and object_ or None
            c = (context is not None and (not isinstance(context.identifier, REGEXTerm))) and context or None
            for (s1, p1, o1), cg in self.storage.triples((s, p, o), c):
                matchingCtxs = []
                for ctx in cg:
                    if c is None:
                        if context is None or context.identifier.compiledExpr.match(ctx.identifier):
                            matchingCtxs.append(ctx)
                    else:
                        matchingCtxs.append(ctx)
                if matchingCtxs and regexCompareQuad((s1, p1, o1, None), (subject, predicate, object_, None)):
                    yield ((s1, p1, o1), (c for c in matchingCtxs))
        else:
            for (s1, p1, o1), cg in self.storage.triples((subject, predicate, object_), context):
                yield ((s1, p1, o1), cg)

    def __len__(self, context=None):
        return self.storage.__len__(context)

    def contexts(self, triple=None):
        for ctx in self.storage.contexts(triple):
            yield ctx

    def remove_context(self, identifier):
        self.storage.remove((None, None, None), identifier)

    def bind(self, prefix, namespace, override=True):
        self.storage.bind(prefix, namespace, override=override)

    def prefix(self, namespace):
        return self.storage.prefix(namespace)

    def namespace(self, prefix):
        return self.storage.namespace(prefix)

    def namespaces(self):
        return self.storage.namespaces()

    def commit(self):
        self.storage.commit()

    def rollback(self):
        self.storage.rollback()