import warnings
from transformers import AutoTokenizer
from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum
from ...processing_utils import ProcessorMixin
class DecodeType(ExplicitEnum):
    CHARACTER = 'char'
    BPE = 'bpe'
    WORDPIECE = 'wp'