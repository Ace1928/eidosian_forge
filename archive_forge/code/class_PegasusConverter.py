import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class PegasusConverter(SpmConverter):

    def vocab(self, proto):
        vocab = [(self.original_tokenizer.pad_token, 0.0), (self.original_tokenizer.eos_token, 0.0)]
        if self.original_tokenizer.mask_token_sent is not None:
            vocab += [(self.original_tokenizer.mask_token_sent, 0.0)]
        if self.original_tokenizer.mask_token is not None and self.original_tokenizer.mask_token_id < self.original_tokenizer.offset:
            vocab += [(self.original_tokenizer.mask_token, 0.0)]
        vocab += [(f'<unk_{i}>', -100.0) for i in range(2, self.original_tokenizer.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.original_tokenizer.offset

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)])

    def post_processor(self):
        eos = self.original_tokenizer.eos_token
        special_tokens = [(eos, self.original_tokenizer.eos_token_id)]
        return processors.TemplateProcessing(single=['$A', eos], pair=['$A', '$B', eos], special_tokens=special_tokens)