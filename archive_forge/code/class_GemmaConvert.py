import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class GemmaConvert(SpmConverter):
    handle_byte_fallback = True
    '"\n    split_by_unicode_script: true\n    split_by_number: true\n    split_by_whitespace: true\n    treat_whitespace_as_suffix: false\n    allow_whitespace_only_pieces: true\n    split_digits: true\n    byte_fallback: true\n    '

    def normalizer(self, proto):
        return normalizers.Replace(' ', '▁')

    def vocab(self, proto):
        vocab = [(self.original_tokenizer.pad_token, 0.0), (self.original_tokenizer.eos_token, 0.0), (self.original_tokenizer.bos_token, 0.0)]
        for piece in proto.pieces[3:]:
            if piece.piece == '<0x09>':
                vocab += [('\t', piece.score)]
            else:
                vocab += [(piece.piece, piece.score)]
        return vocab

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence([decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse()])

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:
            import tokenizers
            if version.parse(tokenizers.__version__) < version.parse('0.14.0'):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))
        elif model_type == 2:
            _, merges = GemmaSentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True, dropout=None))
            tokenizer.add_special_tokens([AddedToken('<pad>', normalized=False, special=True), AddedToken('<eos>', normalized=False, special=True), AddedToken('<bos>', normalized=False, special=True), AddedToken('<unk>', normalized=False, special=True)])
        else:
            raise Exception("You're trying to run a `Unigram` model but you're file was trained with a different algorithm")
        return tokenizer