import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class BlenderbotConverter(Converter):

    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(single=f'$A:0 {ot.eos_token}:0', special_tokens=[(ot.eos_token, ot.eos_token_id)])
        return tokenizer