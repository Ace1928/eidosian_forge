from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
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