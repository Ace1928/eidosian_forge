import re
from wasabi import Printer
from ...tokens import Doc, Span, Token
from ...training import biluo_tags_to_spans, iob_to_biluo
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
Create an Example from the lines for one CoNLL-U sentence, merging
    subtokens and appending morphology to tags if required.

    lines (str): The non-comment lines for a CoNLL-U sentence
    ner_tag_pattern (str): The regex pattern for matching NER in MISC col
    RETURNS (Example): An example containing the annotation
    