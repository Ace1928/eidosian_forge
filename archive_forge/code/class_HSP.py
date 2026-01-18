import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
class HSP(_BaseHSP):
    """Class representing high-scoring region(s) between query and hit.

    HSP (high-scoring pair) objects are contained by Hit objects (see Hit).
    In most cases, HSP objects store the bulk of the statistics and results
    (e.g. e-value, bitscores, query sequence, etc.) produced by a search
    program.

    Depending on the search output file format, a given HSP will contain one
    or more HSPFragment object(s). Examples of search programs that produce HSP
    with one HSPFragments are BLAST, HMMER, and FASTA. Other programs such as
    BLAT or Exonerate may produce HSPs containing more than one HSPFragment.
    However, their native terminologies may differ: in BLAT these fragments
    are called 'blocks' while in Exonerate they are called exons or NER.

    Here are examples from each type of HSP. The first one comes from a BLAST
    search::

        >>> from Bio import SearchIO
        >>> blast_qresult = next(SearchIO.parse('Blast/mirna.xml', 'blast-xml'))
        >>> blast_hsp = blast_qresult[1][0]     # the first HSP from the second hit
        >>> blast_hsp
        HSP(hit_id='gi|301171311|ref|NR_035856.1|', query_id='33211', 1 fragments)
        >>> print(blast_hsp)
              Query: 33211 mir_1
                Hit: gi|301171311|ref|NR_035856.1| Pan troglodytes microRNA mir-520b ...
        Query range: [1:61] (1)
          Hit range: [0:60] (1)
        Quick stats: evalue 1.7e-22; bitscore 109.49
          Fragments: 1 (60 columns)
             Query - CCTCTACAGGGAAGCGCTTTCTGTTGTCTGAAAGAAAAGAAAGTGCTTCCTTTTAGAGGG
                     ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
               Hit - CCTCTACAGGGAAGCGCTTTCTGTTGTCTGAAAGAAAAGAAAGTGCTTCCTTTTAGAGGG

    For HSPs with a single HSPFragment, you can invoke ``print`` on it and see the
    underlying sequence alignment, if it exists. This is not the case for HSPs
    with more than one HSPFragment. Below is an example, using an HSP from a
    BLAT search. Invoking ``print`` on these HSPs will instead show a table of the
    HSPFragment objects it contains::

        >>> blat_qresult = SearchIO.read('Blat/mirna.pslx', 'blat-psl', pslx=True)
        >>> blat_hsp = blat_qresult[1][0]       # the first HSP from the second hit
        >>> blat_hsp
        HSP(hit_id='chr11', query_id='blat_1', 2 fragments)
        >>> print(blat_hsp)
              Query: blat_1 <unknown description>
                Hit: chr11 <unknown description>
        Query range: [42:67] (-1)
          Hit range: [59018929:59018955] (1)
        Quick stats: evalue ?; bitscore ?
          Fragments: ---  --------------  ----------------------  ----------------------
                       #            Span             Query range               Hit range
                     ---  --------------  ----------------------  ----------------------
                       0               6                 [61:67]     [59018929:59018935]
                       1              16                 [42:58]     [59018939:59018955]

    Notice that in HSPs with more than one HSPFragments, the HSP's ``query_range``
    ``hit_range`` properties encompasses all fragments it contains.

    You can check whether an HSP has more than one HSPFragments or not using the
    ``is_fragmented`` property::

        >>> blast_hsp.is_fragmented
        False
        >>> blat_hsp.is_fragmented
        True

    Since HSP objects are also containers similar to Python lists, you can
    access a single fragment in an HSP using its integer index::

        >>> blat_fragment = blat_hsp[0]
        >>> print(blat_fragment)
              Query: blat_1 <unknown description>
                Hit: chr11 <unknown description>
        Query range: [61:67] (-1)
          Hit range: [59018929:59018935] (1)
          Fragments: 1 (6 columns)
             Query - tatagt
               Hit - tatagt

    This applies to HSPs objects with a single fragment as well::

        >>> blast_fragment = blast_hsp[0]
        >>> print(blast_fragment)
              Query: 33211 mir_1
                Hit: gi|301171311|ref|NR_035856.1| Pan troglodytes microRNA mir-520b ...
        Query range: [1:61] (1)
          Hit range: [0:60] (1)
          Fragments: 1 (60 columns)
             Query - CCTCTACAGGGAAGCGCTTTCTGTTGTCTGAAAGAAAAGAAAGTGCTTCCTTTTAGAGGG
                     ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
               Hit - CCTCTACAGGGAAGCGCTTTCTGTTGTCTGAAAGAAAAGAAAGTGCTTCCTTTTAGAGGG

    Regardless of the search output file format, HSP objects provide the
    properties listed below. These properties always return values in a list,
    due to the HSP object itself being a list-like container. However, for
    HSP objects with a single HSPFragment, shortcut properties that fetches
    the item from the list are also provided.

    +----------------------+---------------------+-----------------------------+
    | Property             | Shortcut            | Value                       |
    +======================+=====================+=============================+
    | aln_all              | aln                 | HSP alignments as           |
    |                      |                     | MultipleSeqAlignment object |
    +----------------------+---------------------+-----------------------------+
    | aln_annotation_all   | aln_annotation      | dictionary of annotation(s) |
    |                      |                     | of all fragments' alignments|
    +----------------------+---------------------+-----------------------------+
    | fragments            | fragment            | HSPFragment objects         |
    +----------------------+---------------------+-----------------------------+
    | hit_all              | hit                 | hit sequence as SeqRecord   |
    |                      |                     | objects                     |
    +----------------------+---------------------+-----------------------------+
    | hit_features_all     | hit_features        | SeqFeatures of all hit      |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | hit_start_all        | hit_start*          | start coordinates of the    |
    |                      |                     | hit fragments               |
    +----------------------+---------------------+-----------------------------+
    | hit_end_all          | hit_end*            | end coordinates of the hit  |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | hit_span_all         | hit_span*           | sizes of each hit fragments |
    +----------------------+---------------------+-----------------------------+
    | hit_strand_all       | hit_strand          | strand orientations of the  |
    |                      |                     | hit fragments               |
    +----------------------+---------------------+-----------------------------+
    | hit_frame_all        | hit_frame           | reading frames of the hit   |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | hit_range_all        | hit_range           | tuples of start and end     |
    |                      |                     | coordinates of each hit     |
    |                      |                     | fragment                    |
    +----------------------+---------------------+-----------------------------+
    | query_all            | query               | query sequence as SeqRecord |
    |                      |                     | object                      |
    +----------------------+---------------------+-----------------------------+
    | query_features_all   | query_features      | SeqFeatures of all query    |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | query_start_all      | query_start*        | start coordinates of the    |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | query_end_all        | query_end*          | end coordinates of the      |
    |                      |                     | query fragments             |
    +----------------------+---------------------+-----------------------------+
    | query_span_all       | query_span*         | sizes of each query         |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | query_strand_all     | query_strand        | strand orientations of the  |
    |                      |                     | query fragments             |
    +----------------------+---------------------+-----------------------------+
    | query_frame_all      | query_frame         | reading frames of the query |
    |                      |                     | fragments                   |
    +----------------------+---------------------+-----------------------------+
    | query_range_all      | query_range         | tuples of start and end     |
    |                      |                     | coordinates of each query   |
    |                      |                     | fragment                    |
    +----------------------+---------------------+-----------------------------+

    For all types of HSP objects, the property will return the values in a list.
    Shortcuts are only applicable for HSPs with one fragment. Except the ones
    noted, if they are used on an HSP with more than one fragments, an exception
    will be raised.

    For properties that may be used in HSPs with multiple or single fragments
    (``*_start``, ``*_end``, and ``*_span`` properties), their interpretation depends
    on how many fragment the HSP has:

    +------------+---------------------------------------------------+
    | Property   | Value                                             |
    +============+===================================================+
    | hit_start  | smallest coordinate value of all hit fragments    |
    +------------+---------------------------------------------------+
    | hit_end    | largest coordinate value of all hit fragments     |
    +------------+---------------------------------------------------+
    | hit_span   | difference between ``hit_start`` and ``hit_end``  |
    +------------+---------------------------------------------------+
    | query_start| smallest coordinate value of all query fragments  |
    +------------+---------------------------------------------------+
    | query_end  | largest coordinate value of all query fragments   |
    +------------+---------------------------------------------------+
    | query_span | difference between ``query_start`` and            |
    |            | ``query_end``                                     |
    +------------+---------------------------------------------------+

    In addition to the objects listed above, HSP objects also provide the
    following properties and/or attributes:

    +--------------------+------------------------------------------------------+
    | Property           | Value                                                |
    +====================+======================================================+
    | aln_span           | total number of residues in all HSPFragment objects  |
    +--------------------+------------------------------------------------------+
    | molecule_type      | molecule_type of the hit and query SeqRecord objects |
    +--------------------+------------------------------------------------------+
    | is_fragmented      | boolean, whether there are multiple fragments or not |
    +--------------------+------------------------------------------------------+
    | hit_id             | ID of the hit sequence                               |
    +--------------------+------------------------------------------------------+
    | hit_description    | description of the hit sequence                      |
    +--------------------+------------------------------------------------------+
    | hit_inter_ranges   | list of hit sequence coordinates of the regions      |
    |                    | between fragments                                    |
    +--------------------+------------------------------------------------------+
    | hit_inter_spans    | list of lengths of the regions between hit fragments |
    +--------------------+------------------------------------------------------+
    | output_index       | 0-based index for storing the order by which the HSP |
    |                    | appears in the output file (default: -1).            |
    +--------------------+------------------------------------------------------+
    | query_id           | ID of the query sequence                             |
    +--------------------+------------------------------------------------------+
    | query_description  | description of the query sequence                    |
    +--------------------+------------------------------------------------------+
    | query_inter_ranges | list of query sequence coordinates of the regions    |
    |                    | between fragments                                    |
    +--------------------+------------------------------------------------------+
    | query_inter_spans  | list of lengths of the regions between query         |
    |                    | fragments                                            |
    +--------------------+------------------------------------------------------+

    .. [1] may be used in HSPs with multiple fragments

    """
    _NON_STICKY_ATTRS = ('_items',)

    def __init__(self, fragments=(), output_index=-1):
        """Initialize an HSP object.

        :param fragments: fragments contained in the HSP object
        :type fragments: iterable yielding HSPFragment
        :param output_index: optional index / ordering of the HSP fragment in
            the original input file.
        :type output_index: integer

        HSP objects must be initialized with a list containing at least one
        HSPFragment object. If multiple HSPFragment objects are used for
        initialization, they must all have the same ``query_id``,
        ``query_description``, ``hit_id``, ``hit_description``, and
        ``molecule_type`` properties.

        """
        if not fragments:
            raise ValueError('HSP objects must have at least one HSPFragment object.')
        for attr in ('query_id', 'query_description', 'hit_id', 'hit_description', 'molecule_type'):
            if len({getattr(frag, attr) for frag in fragments}) != 1:
                raise ValueError('HSP object can not contain fragments with more than one %s.' % attr)
        self.output_index = output_index
        self._items = []
        for fragment in fragments:
            self._validate_fragment(fragment)
            self._items.append(fragment)

    def __repr__(self):
        """Return string representation of HSP object."""
        return '%s(hit_id=%r, query_id=%r, %r fragments)' % (self.__class__.__name__, self.hit_id, self.query_id, len(self))

    def __iter__(self):
        """Iterate over HSP items."""
        return iter(self._items)

    def __contains__(self, fragment):
        """Return True if HSPFragment is on HSP items."""
        return fragment in self._items

    def __len__(self):
        """Return number of HSPs items."""
        return len(self._items)

    def __bool__(self):
        """Return True if it has HSPs."""
        return bool(self._items)

    def __str__(self):
        """Return a human readable summary of the HSP object."""
        lines = []
        statline = []
        evalue = getattr_str(self, 'evalue', fmt='%.2g')
        statline.append('evalue ' + evalue)
        bitscore = getattr_str(self, 'bitscore', fmt='%.2f')
        statline.append('bitscore ' + bitscore)
        lines.append('Quick stats: ' + '; '.join(statline))
        if len(self.fragments) == 1:
            return '\n'.join([self._str_hsp_header(), '\n'.join(lines), self.fragments[0]._str_aln()])
        else:
            lines.append('  Fragments: %s  %s  %s  %s' % ('-' * 3, '-' * 14, '-' * 22, '-' * 22))
            pattern = '%16s  %14s  %22s  %22s'
            lines.append(pattern % ('#', 'Span', 'Query range', 'Hit range'))
            lines.append(pattern % ('-' * 3, '-' * 14, '-' * 22, '-' * 22))
            for idx, block in enumerate(self.fragments):
                aln_span = getattr_str(block, 'aln_span')
                query_start = getattr_str(block, 'query_start')
                query_end = getattr_str(block, 'query_end')
                query_range = '[%s:%s]' % (query_start, query_end)
                query_range = query_range[:20] + '~]' if len(query_range) > 22 else query_range
                hit_start = getattr_str(block, 'hit_start')
                hit_end = getattr_str(block, 'hit_end')
                hit_range = '[%s:%s]' % (hit_start, hit_end)
                hit_range = hit_range[:20] + '~]' if len(hit_range) > 22 else hit_range
                lines.append(pattern % (str(idx), aln_span, query_range, hit_range))
            return self._str_hsp_header() + '\n' + '\n'.join(lines)

    def __getitem__(self, idx):
        """Return object of index idx."""
        if isinstance(idx, slice):
            obj = self.__class__(self._items[idx])
            self._transfer_attrs(obj)
            return obj
        return self._items[idx]

    def __setitem__(self, idx, fragments):
        """Set an item of index idx with the given fragments."""
        if isinstance(fragments, (list, tuple)):
            for fragment in fragments:
                self._validate_fragment(fragment)
        else:
            self._validate_fragment(fragments)
        self._items[idx] = fragments

    def __delitem__(self, idx):
        """Delete item of index idx."""
        del self._items[idx]

    def _validate_fragment(self, fragment):
        if not isinstance(fragment, HSPFragment):
            raise TypeError('HSP objects can only contain HSPFragment objects.')
        if self._items:
            if fragment.hit_id != self.hit_id:
                raise ValueError('Expected HSPFragment with hit ID %r, found %r instead.' % (self.id, fragment.hit_id))
            if fragment.hit_description != self.hit_description:
                raise ValueError('Expected HSPFragment with hit description %r, found %r instead.' % (self.description, fragment.hit_description))
            if fragment.query_id != self.query_id:
                raise ValueError('Expected HSPFragment with query ID %r, found %r instead.' % (self.query_id, fragment.query_id))
            if fragment.query_description != self.query_description:
                raise ValueError('Expected HSP with query description %r, found %r instead.' % (self.query_description, fragment.query_description))

    def _aln_span_get(self):
        return sum((frg.aln_span for frg in self.fragments))
    aln_span = property(fget=_aln_span_get, doc='Total number of columns in all HSPFragment objects.')

    def _get_coords(self, seq_type, coord_type):
        assert seq_type in ('hit', 'query')
        assert coord_type in ('start', 'end')
        coord_name = '%s_%s' % (seq_type, coord_type)
        coords = [getattr(frag, coord_name) for frag in self.fragments]
        if None in coords:
            warnings.warn("'None' exist in %s coordinates; ignored" % coord_name, BiopythonWarning)
        return coords

    def _hit_start_get(self):
        return min(self._get_coords('hit', 'start'))
    hit_start = property(fget=_hit_start_get, doc='Smallest coordinate value of all hit fragments.')

    def _query_start_get(self):
        return min(self._get_coords('query', 'start'))
    query_start = property(fget=_query_start_get, doc='Smallest coordinate value of all query fragments.')

    def _hit_end_get(self):
        return max(self._get_coords('hit', 'end'))
    hit_end = property(fget=_hit_end_get, doc='Largest coordinate value of all hit fragments.')

    def _query_end_get(self):
        return max(self._get_coords('query', 'end'))
    query_end = property(fget=_query_end_get, doc='Largest coordinate value of all hit fragments.')

    def _hit_span_get(self):
        try:
            return self.hit_end - self.hit_start
        except TypeError:
            return None
    hit_span = property(fget=_hit_span_get, doc='The number of hit residues covered by the HSP.')

    def _query_span_get(self):
        try:
            return self.query_end - self.query_start
        except TypeError:
            return None
    query_span = property(fget=_query_span_get, doc='The number of query residues covered by the HSP.')

    def _hit_range_get(self):
        return (self.hit_start, self.hit_end)
    hit_range = property(fget=_hit_range_get, doc='Tuple of HSP hit start and end coordinates.')

    def _query_range_get(self):
        return (self.query_start, self.query_end)
    query_range = property(fget=_query_range_get, doc='Tuple of HSP query start and end coordinates.')

    def _inter_ranges_get(self, seq_type):
        assert seq_type in ('query', 'hit')
        strand = getattr(self, '%s_strand_all' % seq_type)[0]
        coords = getattr(self, '%s_range_all' % seq_type)
        if strand == -1:
            startfunc, endfunc = (min, max)
        else:
            startfunc, endfunc = (max, min)
        inter_coords = []
        for idx, coord in enumerate(coords[:-1]):
            start = startfunc(coords[idx])
            end = endfunc(coords[idx + 1])
            inter_coords.append((min(start, end), max(start, end)))
        return inter_coords

    def _hit_inter_ranges_get(self):
        return self._inter_ranges_get('hit')
    hit_inter_ranges = property(fget=_hit_inter_ranges_get, doc='Hit sequence coordinates of the regions between fragments.')

    def _query_inter_ranges_get(self):
        return self._inter_ranges_get('query')
    query_inter_ranges = property(fget=_query_inter_ranges_get, doc='Query sequence coordinates of the regions between fragments.')

    def _inter_spans_get(self, seq_type):
        assert seq_type in ('query', 'hit')
        attr_name = '%s_inter_ranges' % seq_type
        return [coord[1] - coord[0] for coord in getattr(self, attr_name)]

    def _hit_inter_spans_get(self):
        return self._inter_spans_get('hit')
    hit_inter_spans = property(fget=_hit_inter_spans_get, doc='Lengths of regions between hit fragments.')

    def _query_inter_spans_get(self):
        return self._inter_spans_get('query')
    query_inter_spans = property(fget=_query_inter_spans_get, doc='Lengths of regions between query fragments.')
    is_fragmented = property(lambda self: len(self) > 1, doc='Whether the HSP has more than one HSPFragment objects.')
    hit_description = fullcascade('hit_description', doc='Description of the hit sequence.')
    query_description = fullcascade('query_description', doc='Description of the query sequence.')
    hit_id = fullcascade('hit_id', doc='ID of the hit sequence.')
    query_id = fullcascade('query_id', doc='ID of the query sequence.')
    molecule_type = fullcascade('molecule_type', doc='molecule_type of the hit and query SeqRecord objects.')
    fragment = singleitem(doc='HSPFragment object, first fragment.')
    hit = singleitem('hit', doc='Hit sequence as a SeqRecord object, first fragment.')
    query = singleitem('query', doc='Query sequence as a SeqRecord object, first fragment.')
    aln = singleitem('aln', doc='Alignment of the first fragment as a MultipleSeqAlignment object.')
    aln_annotation = singleitem('aln_annotation', doc="Dictionary of annotation(s) of the first fragment's alignment.")
    hit_features = singleitem('hit_features', doc='Hit sequence features, first fragment.')
    query_features = singleitem('query_features', doc='Query sequence features, first fragment.')
    hit_strand = singleitem('hit_strand', doc='Hit strand orientation, first fragment.')
    query_strand = singleitem('query_strand', doc='Query strand orientation, first fragment.')
    hit_frame = singleitem('hit_frame', doc='Hit sequence reading frame, first fragment.')
    query_frame = singleitem('query_frame', doc='Query sequence reading frame, first fragment.')
    fragments = allitems(doc='List of all HSPFragment objects.')
    hit_all = allitems('hit', doc="List of all fragments' hit sequences as SeqRecord objects.")
    query_all = allitems('query', doc="List of all fragments' query sequences as SeqRecord objects.")
    aln_all = allitems('aln', doc="List of all fragments' alignments as MultipleSeqAlignment objects.")
    aln_annotation_all = allitems('aln_annotation', doc="Dictionary of annotation(s) of all fragments' alignments.")
    hit_features_all = allitems('hit_features', doc='List of all hit sequence features.')
    query_features_all = allitems('query_features', doc='List of all query sequence features.')
    hit_strand_all = allitems('hit_strand', doc="List of all fragments' hit sequence strands.")
    query_strand_all = allitems('query_strand', doc="List of all fragments' query sequence strands")
    hit_frame_all = allitems('hit_frame', doc="List of all fragments' hit sequence reading frames.")
    query_frame_all = allitems('query_frame', doc="List of all fragments' query sequence reading frames.")
    hit_start_all = allitems('hit_start', doc="List of all fragments' hit start coordinates.")
    query_start_all = allitems('query_start', doc="List of all fragments' query start coordinates.")
    hit_end_all = allitems('hit_end', doc="List of all fragments' hit end coordinates.")
    query_end_all = allitems('query_end', doc="List of all fragments' query end coordinates.")
    hit_span_all = allitems('hit_span', doc="List of all fragments' hit sequence size.")
    query_span_all = allitems('query_span', doc="List of all fragments' query sequence size.")
    hit_range_all = allitems('hit_range', doc="List of all fragments' hit start and end coordinates.")
    query_range_all = allitems('query_range', doc="List of all fragments' query start and end coordinates.")