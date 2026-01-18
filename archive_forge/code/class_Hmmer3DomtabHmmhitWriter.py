from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabHmmhitWriter:
    """HMMER domain table writer using hit coordinates.

    Writer for hmmer3-domtab output format which writes hit coordinates
    as HMM profile coordinates.
    """
    hmm_as_hit = True

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
        """Return the header string of a domain HMMER table output (PRIVATE)."""
        if first_qresult:
            qnamew = 20
            tnamew = max(20, len(first_qresult[0].id))
            try:
                qaccw = max(10, len(first_qresult.acc))
                taccw = max(10, len(first_qresult[0].acc))
            except AttributeError:
                qaccw, taccw = (10, 10)
        else:
            qnamew, tnamew, qaccw, taccw = (20, 20, 10, 10)
        header = '#%*s %22s %40s %11s %11s %11s\n' % (tnamew + qnamew - 1 + 15 + taccw + qaccw, '', '--- full sequence ---', '-------------- this domain -------------', 'hmm coord', 'ali coord', 'env coord')
        header += '#%-*s %-*s %5s %-*s %-*s %5s %9s %6s %5s %3s %3s %9s %9s %6s %5s %5s %5s %5s %5s %5s %5s %4s %s\n' % (tnamew - 1, ' target name', taccw, 'accession', 'tlen', qnamew, 'query name', qaccw, 'accession', 'qlen', 'E-value', 'score', 'bias', '#', 'of', 'c-Evalue', 'i-Evalue', 'score', 'bias', 'from', 'to', 'from', 'to', 'from', 'to', 'acc', 'description of target')
        header += '#%*s %*s %5s %*s %*s %5s %9s %6s %5s %3s %3s %9s %9s %6s %5s %5s %5s %5s %5s %5s %5s %4s %s\n' % (tnamew - 1, '-------------------', taccw, '----------', '-----', qnamew, '--------------------', qaccw, '----------', '-----', '---------', '------', '-----', '---', '---', '---------', '---------', '------', '-----', '-----', '-----', '-----', '-----', '-----', '-----', '----', '---------------------')
        return header

    def _build_row(self, qresult):
        """Return a string or one row or more of the QueryResult object (PRIVATE)."""
        rows = ''
        qnamew = max(20, len(qresult.id))
        tnamew = max(20, len(qresult[0].id))
        try:
            qaccw = max(10, len(qresult.accession))
            taccw = max(10, len(qresult[0].accession))
            qresult_acc = qresult.accession
        except AttributeError:
            qaccw, taccw = (10, 10)
            qresult_acc = '-'
        for hit in qresult:
            try:
                hit_acc = hit.accession
            except AttributeError:
                hit_acc = '-'
            for hsp in hit.hsps:
                if self.hmm_as_hit:
                    hmm_to = hsp.hit_end
                    hmm_from = hsp.hit_start + 1
                    ali_to = hsp.query_end
                    ali_from = hsp.query_start + 1
                else:
                    hmm_to = hsp.query_end
                    hmm_from = hsp.query_start + 1
                    ali_to = hsp.hit_end
                    ali_from = hsp.hit_start + 1
                rows += '%-*s %-*s %5d %-*s %-*s %5d %9.2g %6.1f %5.1f %3d %3d %9.2g %9.2g %6.1f %5.1f %5d %5d %5ld %5ld %5d %5d %4.2f %s\n' % (tnamew, hit.id, taccw, hit_acc, hit.seq_len, qnamew, qresult.id, qaccw, qresult_acc, qresult.seq_len, hit.evalue, hit.bitscore, hit.bias, hsp.domain_index, len(hit.hsps), hsp.evalue_cond, hsp.evalue, hsp.bitscore, hsp.bias, hmm_from, hmm_to, ali_from, ali_to, hsp.env_start + 1, hsp.env_end, hsp.acc_avg, hit.description)
        return rows