import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class CamembertConverter(SpmConverter):

    def vocab(self, proto):
        vocab = [('<s>NOTUSED', 0.0), ('<pad>', 0.0), ('</s>NOTUSED', 0.0), ('<unk>', 0.0), ('<unk>NOTUSED', -100)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[1:]]
        vocab += [('<mask>', 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>', special_tokens=[('<s>', self.original_tokenizer.convert_tokens_to_ids('<s>')), ('</s>', self.original_tokenizer.convert_tokens_to_ids('</s>'))])