import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
class GemmaSentencePieceExtractor(SentencePieceExtractor):

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        vocab['\t'] = vocab.pop('<0x09>')
        if vocab_scores is not None:
            vocab_scores, reverse = (dict(vocab_scores), True)
        else:
            vocab_scores, reverse = (vocab, False)
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = (merge[:index], merge[index:])
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        return (vocab, merges)