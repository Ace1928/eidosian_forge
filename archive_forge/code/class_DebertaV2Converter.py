import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class DebertaV2Converter(SpmConverter):

    def pre_tokenizer(self, replacement, add_prefix_space):
        list_pretokenizers = []
        if self.original_tokenizer.split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior='isolated'))
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space))
        return pre_tokenizers.Sequence(list_pretokenizers)

    def normalizer(self, proto):
        list_normalizers = []
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        list_normalizers.append(normalizers.Strip())
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(' {2,}'), ' '))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(single='[CLS]:0 $A:0 [SEP]:0', pair='[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1', special_tokens=[('[CLS]', self.original_tokenizer.convert_tokens_to_ids('[CLS]')), ('[SEP]', self.original_tokenizer.convert_tokens_to_ids('[SEP]'))])