import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class XLNetConverter(SpmConverter):

    def vocab(self, proto):
        return [(piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100) for piece in proto.pieces]

    def normalizer(self, proto):
        list_normalizers = [normalizers.Replace('``', '"'), normalizers.Replace("''", '"')]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(' {2,}'), ' '))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(single='$A:0 <sep>:0 <cls>:2', pair='$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2', special_tokens=[('<sep>', self.original_tokenizer.convert_tokens_to_ids('<sep>')), ('<cls>', self.original_tokenizer.convert_tokens_to_ids('<cls>'))])