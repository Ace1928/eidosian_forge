from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_location_qualifier_value(adaptor, location_id):
    value = adaptor.execute_and_fetch_col0('SELECT value FROM location_qualifier_value WHERE location_id = %s', (location_id,))
    try:
        return value[0]
    except IndexError:
        return ''