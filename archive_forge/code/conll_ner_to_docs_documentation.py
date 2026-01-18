from wasabi import Printer
from ...errors import Errors
from ...tokens import Doc, Span
from ...training import iob_to_biluo
from ...util import get_lang_class, load_model
from .. import tags_to_entities

    Convert files in the CoNLL-2003 NER format and similar
    whitespace-separated columns into Doc objects.

    The first column is the tokens, the final column is the IOB tags. If an
    additional second column is present, the second column is the tags.

    Sentences are separated with whitespace and documents can be separated
    using the line "-DOCSTART- -X- O O".

    Sample format:

    -DOCSTART- -X- O O

    I O
    like O
    London B-GPE
    and O
    New B-GPE
    York I-GPE
    City I-GPE
    . O

    