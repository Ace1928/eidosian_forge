from itertools import chain
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class Hmmer3TabWriter:
    """Writer for hmmer3-tab output format."""

    def __init__(self, handle):
        """Initialize the class."""
        self.handle = handle

    def write_file(self, qresults):
        """Write to the handle.

        Returns a tuple of how many QueryResult, Hit, and HSP objects were written.

        """
        handle = self.handle
        qresult_counter, hit_counter, hsp_counter, frag_counter = (0, 0, 0, 0)
        try:
            first_qresult = next(qresults)
        except StopIteration:
            handle.write(self._build_header())
        else:
            handle.write(self._build_header(first_qresult))
            for qresult in chain([first_qresult], qresults):
                if qresult:
                    handle.write(self._build_row(qresult))
                    qresult_counter += 1
                    hit_counter += len(qresult)
                    hsp_counter += sum((len(hit) for hit in qresult))
                    frag_counter += sum((len(hit.fragments) for hit in qresult))
        return (qresult_counter, hit_counter, hsp_counter, frag_counter)

    def _build_header(self, first_qresult=None):
        """Return the header string of a HMMER table output (PRIVATE)."""
        if first_qresult is not None:
            qnamew = 20
            tnamew = max(20, len(first_qresult[0].id))
            qaccw = max(10, len(first_qresult.accession))
            taccw = max(10, len(first_qresult[0].accession))
        else:
            qnamew, tnamew, qaccw, taccw = (20, 20, 10, 10)
        header = '#%*s %22s %22s %33s\n' % (tnamew + qnamew + taccw + qaccw + 2, '', '--- full sequence ----', '--- best 1 domain ----', '--- domain number estimation ----')
        header += '#%-*s %-*s %-*s %-*s %9s %6s %5s %9s %6s %5s %5s %3s %3s %3s %3s %3s %3s %3s %s\n' % (tnamew - 1, ' target name', taccw, 'accession', qnamew, 'query name', qaccw, 'accession', '  E-value', ' score', ' bias', '  E-value', ' score', ' bias', 'exp', 'reg', 'clu', ' ov', 'env', 'dom', 'rep', 'inc', 'description of target')
        header += '#%*s %*s %*s %*s %9s %6s %5s %9s %6s %5s %5s %3s %3s %3s %3s %3s %3s %3s %s\n' % (tnamew - 1, '-------------------', taccw, '----------', qnamew, '--------------------', qaccw, '----------', '---------', '------', '-----', '---------', '------', '-----', '---', '---', '---', '---', '---', '---', '---', '---', '---------------------')
        return header

    def _build_row(self, qresult):
        """Return a string or one row or more of the QueryResult object (PRIVATE)."""
        rows = ''
        qnamew = max(20, len(qresult.id))
        tnamew = max(20, len(qresult[0].id))
        qaccw = max(10, len(qresult.accession))
        taccw = max(10, len(qresult[0].accession))
        for hit in qresult:
            rows += '%-*s %-*s %-*s %-*s %9.2g %6.1f %5.1f %9.2g %6.1f %5.1f %5.1f %3d %3d %3d %3d %3d %3d %3d %s\n' % (tnamew, hit.id, taccw, hit.accession, qnamew, qresult.id, qaccw, qresult.accession, hit.evalue, hit.bitscore, hit.bias, hit.hsps[0].evalue, hit.hsps[0].bitscore, hit.hsps[0].bias, hit.domain_exp_num, hit.region_num, hit.cluster_num, hit.overlap_num, hit.env_num, hit.domain_obs_num, hit.domain_reported_num, hit.domain_included_num, hit.description)
        return rows