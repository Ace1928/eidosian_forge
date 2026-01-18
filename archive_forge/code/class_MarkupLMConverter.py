import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class MarkupLMConverter(Converter):

    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, dropout=None, continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False, unk_token=self.original_tokenizer.unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        tokenizer.post_processor = processors.TemplateProcessing(single=f'{cls} $A {sep}', pair=f'{cls} $A {sep} $B {sep}', special_tokens=[(cls, cls_token_id), (sep, sep_token_id)])
        return tokenizer