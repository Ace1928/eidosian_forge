import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class SpmConverter(Converter):

    def __init__(self, *args):
        requires_backends(self, 'protobuf')
        super().__init__(*args)
        model_pb2 = import_protobuf()
        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.vocab_file, 'rb') as f:
            m.ParseFromString(f.read())
        self.proto = m
        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, 'handle_byte_fallback', None):
                warnings.warn('The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers. In practice this means that the fast version of the tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these unknown tokens into a sequence of byte tokens matching the original piece of text.')

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)
        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True))
        else:
            raise Exception("You're trying to run a `Unigram` model but you're file was trained with a different algorithm")
        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [normalizers.Strip(left=False, right=True), normalizers.Replace(Regex(' {2,}'), '▁')]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = 'always'
        if hasattr(self.original_tokenizer, 'legacy') and (not self.original_tokenizer.legacy):
            prepend_scheme = 'first'
        return pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space, prepend_scheme=prepend_scheme)

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer
        replacement = '▁'
        add_prefix_space = True
        if hasattr(self.original_tokenizer, 'add_prefix_space'):
            add_prefix_space = self.original_tokenizer.add_prefix_space
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor
        return tokenizer