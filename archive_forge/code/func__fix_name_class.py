from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _fix_name_class(self, entrez_name):
    """Map Entrez name terms to those used in taxdump (PRIVATE).

        We need to make this conversion to match the taxon_name.name_class
        values used by the BioSQL load_ncbi_taxonomy.pl script.

        e.g.::

            "ScientificName" -> "scientific name",
            "EquivalentName" -> "equivalent name",
            "Synonym" -> "synonym",

        """

    def add_space(letter):
        """Add a space before a capital letter."""
        if letter.isupper():
            return ' ' + letter.lower()
        else:
            return letter
    answer = ''.join((add_space(letter) for letter in entrez_name)).strip()
    if answer != answer.lower():
        raise ValueError(f"Expected processed entrez_name, '{answer}' to only have lower case letters.")
    return answer