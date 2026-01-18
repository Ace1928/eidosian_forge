import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous
KEGG link - find related entries by using database cross-references.

    target_db - Target database
    source_db_or_dbentries - source database
    option - Can be "turtle" or "n-triple" (string).
    