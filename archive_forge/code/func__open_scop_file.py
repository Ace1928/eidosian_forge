import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def _open_scop_file(scop_dir_path, version, filetype):
    filename = f'dir.{filetype}.scop.txt_{version}'
    handle = open(os.path.join(scop_dir_path, filename))
    return handle