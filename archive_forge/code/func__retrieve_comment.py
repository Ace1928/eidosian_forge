from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_comment(adaptor, primary_id):
    qvs = adaptor.execute_and_fetchall('SELECT comment_text FROM comment WHERE bioentry_id=%s ORDER BY "rank"', (primary_id,))
    comments = [comm[0] for comm in qvs]
    if comments:
        return {'comment': comments}
    else:
        return {}