from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_term_id(self, name, ontology_id=None, definition=None, identifier=None):
    """Get the id that corresponds to a term (PRIVATE).

        This looks through the term table for a the given term. If it
        is not found, a new id corresponding to this term is created.
        In either case, the id corresponding to that term is returned, so
        that you can reference it in another table.

        The ontology_id should be used to disambiguate the term.
        """
    sql = 'SELECT term_id FROM term WHERE name = %s'
    fields = [name]
    if ontology_id:
        sql += ' AND ontology_id = %s'
        fields.append(ontology_id)
    id_results = self.adaptor.execute_and_fetchall(sql, fields)
    if len(id_results) > 1:
        raise ValueError(f'Multiple term ids for {name}: {id_results!r}')
    elif len(id_results) == 1:
        return id_results[0][0]
    else:
        sql = 'INSERT INTO term (name, definition, identifier, ontology_id) VALUES (%s, %s, %s, %s)'
        self.adaptor.execute(sql, (name, definition, identifier, ontology_id))
        return self.adaptor.last_id('term')