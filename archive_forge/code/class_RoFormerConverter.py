import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class RoFormerConverter(Converter):

    def converted(self) -> Tokenizer:
        from .models.roformer.tokenization_utils import JiebaPreTokenizer
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, 'basic_tokenizer'):
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case
        tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=False, strip_accents=strip_accents, lowercase=do_lower_case)
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))
        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        tokenizer.post_processor = processors.TemplateProcessing(single=f'{cls}:0 $A:0 {sep}:0', pair=f'{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1', special_tokens=[(cls, cls_token_id), (sep, sep_token_id)])
        tokenizer.decoder = decoders.WordPiece(prefix='##')
        return tokenizer