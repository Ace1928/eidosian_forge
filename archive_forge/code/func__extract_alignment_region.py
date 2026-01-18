from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _extract_alignment_region(alignment_seq_with_flanking, annotation):
    """Extract alignment region (PRIVATE).

    Helper function for the main parsing code.

    To get the actual pairwise alignment sequences, we must first
    translate the un-gapped sequence based coordinates into positions
    in the gapped sequence (which may have a flanking region shown
    using leading - characters).  To date, I have never seen any
    trailing flanking region shown in the m10 file, but the
    following code should also cope with that.

    Note that this code seems to work fine even when the "sq_offset"
    entries are present as a result of using the -X command line option.
    """
    align_stripped = alignment_seq_with_flanking.strip('-')
    display_start = int(annotation['al_display_start'])
    if int(annotation['al_start']) <= int(annotation['al_stop']):
        start = int(annotation['al_start']) - display_start
        end = int(annotation['al_stop']) - display_start + 1
    else:
        start = display_start - int(annotation['al_start'])
        end = display_start - int(annotation['al_stop']) + 1
    end += align_stripped.count('-')
    if start < 0 or start >= end or end > len(align_stripped):
        raise ValueError('Problem with sequence start/stop,\n%s[%i:%i]\n%s' % (alignment_seq_with_flanking, start, end, annotation))
    return align_stripped[start:end]