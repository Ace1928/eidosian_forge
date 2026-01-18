import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class Phylogeny(PhyloElement, BaseTree.Tree):
    """A phylogenetic tree.

    :Parameters:
        root : Clade
            the root node/clade of this tree
        rooted : bool
            True if this tree is rooted
        rerootable : bool
            True if this tree is rerootable
        branch_length_unit : string
            unit for branch_length values on clades
        name : string
            identifier for this tree, not required to be unique
        id : Id
            unique identifier for this tree
        description : string
            plain-text description
        date : Date
            date for the root node of this tree
        confidences : list
            Confidence objects for this tree
        clade_relations : list
            CladeRelation objects
        sequence_relations : list
            SequenceRelation objects
        properties : list
            Property objects
        other : list
            non-phyloXML elements (type ``Other``)

    """

    def __init__(self, root=None, rooted=True, rerootable=None, branch_length_unit=None, type=None, name=None, id=None, description=None, date=None, confidences=None, clade_relations=None, sequence_relations=None, properties=None, other=None):
        """Initialize values for phylogenetic tree object."""
        assert isinstance(rooted, bool)
        self.root = root
        self.rooted = rooted
        self.rerootable = rerootable
        self.branch_length_unit = branch_length_unit
        self.type = type
        self.name = name
        self.id = id
        self.description = description
        self.date = date
        self.confidences = confidences or []
        self.clade_relations = clade_relations or []
        self.sequence_relations = sequence_relations or []
        self.properties = properties or []
        self.other = other or []

    @classmethod
    def from_tree(cls, tree, **kwargs):
        """Create a new Phylogeny given a Tree (from Newick/Nexus or BaseTree).

        Keyword arguments are the usual ``Phylogeny`` constructor parameters.
        """
        phy = cls(root=Clade.from_clade(tree.root), rooted=tree.rooted, name=tree.name, id=tree.id is not None and Id(str(tree.id)) or None)
        phy.__dict__.update(kwargs)
        return phy

    @classmethod
    def from_clade(cls, clade, **kwargs):
        """Create a new Phylogeny given a Newick or BaseTree Clade object.

        Keyword arguments are the usual ``PhyloXML.Clade`` constructor parameters.
        """
        return Clade.from_clade(clade).to_phylogeny(**kwargs)

    def as_phyloxml(self):
        """Return this tree, a PhyloXML-compatible Phylogeny object.

        Overrides the ``BaseTree`` method.
        """
        return self

    def to_phyloxml_container(self, **kwargs):
        """Create a new Phyloxml object containing just this phylogeny."""
        return Phyloxml(kwargs, phylogenies=[self])

    def to_alignment(self):
        """Construct a MultipleSeqAlignment from the aligned sequences in this tree."""

        def is_aligned_seq(elem):
            if isinstance(elem, Sequence) and elem.mol_seq.is_aligned:
                return True
            return False
        seqs = self._filter_search(is_aligned_seq, 'preorder', True)
        records = (seq.to_seqrecord() for seq in seqs)
        return MultipleSeqAlignment(records)

    @property
    def alignment(self):
        """Construct an Alignment object from the aligned sequences in this tree."""

        def is_aligned_seq(elem):
            if isinstance(elem, Sequence) and elem.mol_seq.is_aligned:
                return True
            return False
        seqs = self._filter_search(is_aligned_seq, 'preorder', True)
        records = []
        lines = []
        for seq in seqs:
            record = seq.to_seqrecord()
            lines.append(str(record.seq))
            record.seq = record.seq.replace('-', '')
            records.append(record)
        if lines:
            coordinates = Alignment.infer_coordinates(lines)
        else:
            coordinates = None
        return Alignment(records, coordinates)

    def _get_confidence(self):
        """Equivalent to self.confidences[0] if there is only 1 value (PRIVATE).

        See Also: ``Clade.confidence``, ``Clade.taxonomy``

        """
        if len(self.confidences) == 0:
            return None
        if len(self.confidences) > 1:
            raise AttributeError('more than 1 confidence value available; use Phylogeny.confidences')
        return self.confidences[0]

    def _set_confidence(self, value):
        if value is None:
            self.confidences = []
            return
        if isinstance(value, (float, int)):
            value = Confidence(value)
        elif not isinstance(value, Confidence):
            raise ValueError('value must be a number or Confidence instance')
        if len(self.confidences) == 0:
            self.confidences.append(value)
        elif len(self.confidences) == 1:
            self.confidences[0] = value
        else:
            raise ValueError('multiple confidence values already exist; use Phylogeny.confidences instead')

    def _del_confidence(self):
        self.confidences = []
    confidence = property(_get_confidence, _set_confidence, _del_confidence)