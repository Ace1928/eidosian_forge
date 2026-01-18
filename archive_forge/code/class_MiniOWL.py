import sys
from rdflib import Namespace
from rdflib import RDF  as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from ..host import MediaTypes
from ..utils import URIOpener
from . import err_outdated_cache
from . import err_unreachable_vocab
from . import err_unparsable_Turtle_vocab
from . import err_unparsable_ntriples_vocab
from . import err_unparsable_rdfa_vocab
from . import err_unrecognised_vocab_type
from .. import VocabReferenceError
from .cache import CachedVocab, xml_application_media_type
from .. import HTTPError, RDFaError
class MiniOWL:
    """
    Class implementing the simple OWL RL Reasoning required by RDFa in managing vocabulary files. This is done via
    a forward chaining process (in the L{closure} method) using a few simple rules as defined by the RDF and the OWL Semantics
    specifications.

    @ivar graph: the graph that has to be expanded
    @ivar added_triples: each cycle collects the triples that are to be added to the graph eventually.
    @type added_triples: a set, to ensure the unicity of triples being added
    """

    def __init__(self, graph, schema_semantics=False):
        self.graph = graph
        self.added_triples = None
        self.schema_semantics = schema_semantics

    def closure(self):
        """
           Generate the closure the graph. This is the real 'core'.

           The processing rules store new triples via the L{separate method<store_triple>} which stores
           them in the L{added_triples<added_triples>} array. If that array is emtpy at the end of a cycle,
           it means that the whole process can be stopped.
        """
        new_cycle = True
        cycle_num = 0
        while new_cycle:
            cycle_num += 1
            self.added_triples = set()
            for t in self.graph:
                self.rules(t)
            new_cycle = len(self.added_triples) > 0
            for t in self.added_triples:
                self.graph.add(t)

    def store_triple(self, t):
        """
        In contrast to its name, this does not yet add anything to the graph itself, it just stores the tuple in an
        L{internal set<added_triples>}. (It is important for this to be a set: some of the rules in the various closures may
        generate the same tuples several times.) Before adding the tuple to the set, the method checks whether
        the tuple is in the final graph already (if yes, it is not added to the set).

        The set itself is emptied at the start of every processing cycle; the triples are then effectively added to the
        graph at the end of such a cycle. If the set is
        actually empty at that point, this means that the cycle has not added any new triple, and the full processing can stop.

        @param t: the triple to be added to the graph, unless it is already there
        @type t: a 3-element tuple of (s,p,o)
        """
        if t not in self.graph:
            self.added_triples.add(t)

    def rules(self, t):
        """
            Go through the OWL-RL entailement rules prp-spo1, prp-eqp1, prp-eqp2, cax-sco, cax-eqc1, and cax-eqc2 by extending the graph.
            @param t: a triple (in the form of a tuple)
        """
        s, p, o = t
        if self.schema_semantics:
            if p == subPropertyOf:
                for _z, _y, xxx in self.graph.triples((o, subPropertyOf, None)):
                    self.store_triple((s, subPropertyOf, xxx))
            elif p == equivalentProperty:
                for _z, _y, xxx in self.graph.triples((o, equivalentProperty, None)):
                    self.store_triple((s, equivalentProperty, xxx))
                for xxx, _y, _z in self.graph.triples((None, equivalentProperty, s)):
                    self.store_triple((xxx, equivalentProperty, o))
            elif p == subClassOf:
                for _z, _y, xxx in self.graph.triples((o, subClassOf, None)):
                    self.store_triple((s, subClassOf, xxx))
            elif p == equivalentClass:
                for _z, _y, xxx in self.graph.triples((o, equivalentClass, None)):
                    self.store_triple((s, equivalentClass, xxx))
                for xxx, _y, _z in self.graph.triples((None, equivalentClass, s)):
                    self.store_triple((xxx, equivalentClass, o))
        elif p == subPropertyOf:
            for zzz, _z, www in self.graph.triples((None, s, None)):
                self.store_triple((zzz, o, www))
        elif p == equivalentProperty:
            for zzz, _z, www in self.graph.triples((None, s, None)):
                self.store_triple((zzz, o, www))
            for zzz, _z, www in self.graph.triples((None, o, None)):
                self.store_triple((zzz, s, www))
        elif p == subClassOf:
            for vvv, _y, _z in self.graph.triples((None, rdftype, s)):
                self.store_triple((vvv, rdftype, o))
        elif p == equivalentClass:
            for vvv, _y, _z in self.graph.triples((None, rdftype, s)):
                self.store_triple((vvv, rdftype, o))
            for vvv, _y, _z in self.graph.triples((None, rdftype, o)):
                self.store_triple((vvv, rdftype, s))