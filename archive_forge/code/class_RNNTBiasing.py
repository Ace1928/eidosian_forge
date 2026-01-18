import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
class RNNTBiasing(RNNT):
    """torchaudio.models.RNNT()

    Recurrent neural network transducer (RNN-T) model.

    Note:
        To build the model, please use one of the factory functions.

    Args:
        transcriber (torch.nn.Module): transcription network.
        predictor (torch.nn.Module): prediction network.
        joiner (torch.nn.Module): joint network.
        attndim (int): TCPGen attention dimension
        biasing (bool): If true, use biasing, otherwise use standard RNN-T
        deepbiasing (bool): If true, use deep biasing by extracting the biasing vector
        embdim (int): dimension of symbol embeddings
        jointdim (int): dimension of the joint network joint dimension
        charlist (list): The list of word piece tokens in the same order as the output layer
        encoutdim (int): dimension of the encoder output vectors
        dropout_tcpgen (float): dropout rate for TCPGen
        tcpsche (int): The epoch at which TCPGen starts to train
        DBaverage (bool): If true, instead of TCPGen, use DBRNNT for biasing
    """

    def __init__(self, transcriber: _Transcriber, predictor: _Predictor, joiner: _Joiner, attndim: int, biasing: bool, deepbiasing: bool, embdim: int, jointdim: int, charlist: List[str], encoutdim: int, dropout_tcpgen: float, tcpsche: int, DBaverage: bool) -> None:
        super().__init__(transcriber, predictor, joiner)
        self.attndim = attndim
        self.deepbiasing = deepbiasing
        self.jointdim = jointdim
        self.embdim = embdim
        self.encoutdim = encoutdim
        self.char_list = charlist or []
        self.blank_idx = self.char_list.index('<blank>')
        self.nchars = len(self.char_list)
        self.DBaverage = DBaverage
        self.biasing = biasing
        if self.biasing:
            if self.deepbiasing and self.DBaverage:
                self.biasingemb = torch.nn.Linear(self.nchars, self.attndim, bias=False)
            else:
                self.ooKBemb = torch.nn.Embedding(1, self.embdim)
                self.Qproj_char = torch.nn.Linear(self.embdim, self.attndim)
                self.Qproj_acoustic = torch.nn.Linear(self.encoutdim, self.attndim)
                self.Kproj = torch.nn.Linear(self.embdim, self.attndim)
                self.pointer_gate = torch.nn.Linear(self.attndim + self.jointdim, 1)
        self.dropout_tcpgen = torch.nn.Dropout(dropout_tcpgen)
        self.tcpsche = tcpsche

    def forward(self, sources: torch.Tensor, source_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, tries: TrieNode, current_epoch: int, predictor_state: Optional[List[List[torch.Tensor]]]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            tries (TrieNode): wordpiece prefix trees representing the biasing list to be searched
            current_epoch (Int): the current epoch number to determine if TCPGen should be trained
                at this epoch
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
                torch.Tensor
                    TCPGen distribution, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    Generation probability (or copy probability), with shape
                    `(B, max output source length, max output target length, 1)`.
        """
        source_encodings, source_lengths = self.transcriber(input=sources, lengths=source_lengths)
        target_encodings, target_lengths, predictor_state = self.predictor(input=targets, lengths=target_lengths, state=predictor_state)
        hptr = None
        tcpgen_dist, p_gen = (None, None)
        if self.biasing and current_epoch >= self.tcpsche and (tries != []):
            ptrdist_mask, p_gen_mask = self.get_tcpgen_step_masks(targets, tries)
            hptr, tcpgen_dist = self.forward_tcpgen(targets, ptrdist_mask, source_encodings)
            hptr = self.dropout_tcpgen(hptr)
        elif self.biasing:
            if self.DBaverage and self.deepbiasing:
                dummy = self.biasingemb(source_encodings.new_zeros(1, len(self.char_list))).mean()
            else:
                dummy = source_encodings.new_zeros(1, self.embdim)
                dummy = self.Qproj_char(dummy).mean()
                dummy += self.Qproj_acoustic(source_encodings.new_zeros(1, source_encodings.size(-1))).mean()
                dummy += self.Kproj(source_encodings.new_zeros(1, self.embdim)).mean()
                dummy += self.pointer_gate(source_encodings.new_zeros(1, self.attndim + self.jointdim)).mean()
                dummy += self.ooKBemb.weight.mean()
            dummy = dummy * 0
            source_encodings += dummy
        output, source_lengths, target_lengths, jointer_activation = self.joiner(source_encodings=source_encodings, source_lengths=source_lengths, target_encodings=target_encodings, target_lengths=target_lengths, hptr=hptr)
        if self.biasing and hptr is not None and (tcpgen_dist is not None):
            p_gen = torch.sigmoid(self.pointer_gate(torch.cat((jointer_activation, hptr), dim=-1)))
            p_gen = p_gen.masked_fill(p_gen_mask.bool().unsqueeze(1).unsqueeze(-1), 0)
        return (output, source_lengths, target_lengths, predictor_state, tcpgen_dist, p_gen)

    def get_tcpgen_distribution(self, query, ptrdist_mask):
        keyvalues = torch.cat([self.predictor.embedding.weight.data, self.ooKBemb.weight], dim=0)
        keyvalues = self.dropout_tcpgen(self.Kproj(keyvalues))
        tcpgendist = torch.einsum('ntuj,ij->ntui', query, keyvalues)
        tcpgendist = tcpgendist / math.sqrt(query.size(-1))
        ptrdist_mask = ptrdist_mask.unsqueeze(1).repeat(1, tcpgendist.size(1), 1, 1)
        tcpgendist.masked_fill_(ptrdist_mask.bool(), -1000000000.0)
        tcpgendist = torch.nn.functional.softmax(tcpgendist, dim=-1)
        hptr = torch.einsum('ntui,ij->ntuj', tcpgendist[:, :, :, :-1], keyvalues[:-1, :])
        return (hptr, tcpgendist)

    def forward_tcpgen(self, targets, ptrdist_mask, source_encodings):
        tcpgen_dist = None
        if self.DBaverage and self.deepbiasing:
            hptr = self.biasingemb(1 - ptrdist_mask[:, :, :-1].float()).unsqueeze(1)
        else:
            query_char = self.predictor.embedding(targets)
            query_char = self.Qproj_char(query_char).unsqueeze(1)
            query_acoustic = self.Qproj_acoustic(source_encodings).unsqueeze(2)
            query = query_char + query_acoustic
            hptr, tcpgen_dist = self.get_tcpgen_distribution(query, ptrdist_mask)
        return (hptr, tcpgen_dist)

    def get_tcpgen_step_masks(self, yseqs, resettrie):
        seqlen = len(yseqs[0])
        batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
        p_gen_masks = []
        for i, yseq in enumerate(yseqs):
            new_tree = resettrie
            p_gen_mask = []
            for j, vy in enumerate(yseq):
                vy = vy.item()
                new_tree = new_tree[0]
                if vy in [self.blank_idx]:
                    new_tree = resettrie
                    p_gen_mask.append(0)
                elif self.char_list[vy].endswith('▁'):
                    if vy in new_tree and new_tree[vy][0] != {}:
                        new_tree = new_tree[vy]
                    else:
                        new_tree = resettrie
                    p_gen_mask.append(0)
                elif vy not in new_tree:
                    new_tree = [{}]
                    p_gen_mask.append(1)
                else:
                    new_tree = new_tree[vy]
                    p_gen_mask.append(0)
                batch_masks[i, j, list(new_tree[0].keys())] = 0
            p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
        p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()
        return (batch_masks, p_gen_masks)

    def get_tcpgen_step_masks_prefix(self, yseqs, resettrie):
        seqlen = len(yseqs[0])
        batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
        p_gen_masks = []
        for i, yseq in enumerate(yseqs):
            p_gen_mask = []
            new_tree = resettrie
            for j, vy in enumerate(yseq):
                vy = vy.item()
                new_tree = new_tree[0]
                if vy in [self.blank_idx]:
                    new_tree = resettrie
                    batch_masks[i, j, list(new_tree[0].keys())] = 0
                elif self.char_list[vy].startswith('▁'):
                    new_tree = resettrie
                    if vy not in new_tree[0]:
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                    else:
                        new_tree = new_tree[0][vy]
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                        if new_tree[1] != -1:
                            batch_masks[i, j, list(resettrie[0].keys())] = 0
                elif vy not in new_tree:
                    new_tree = resettrie
                    batch_masks[i, j, list(new_tree[0].keys())] = 0
                else:
                    new_tree = new_tree[vy]
                    batch_masks[i, j, list(new_tree[0].keys())] = 0
                    if new_tree[1] != -1:
                        batch_masks[i, j, list(resettrie[0].keys())] = 0
                p_gen_mask.append(0)
            p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
        p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()
        return (batch_masks, p_gen_masks)

    def get_tcpgen_step(self, vy, trie, resettrie):
        new_tree = trie[0]
        if vy in [self.blank_idx]:
            new_tree = resettrie
        elif self.char_list[vy].endswith('▁'):
            if vy in new_tree and new_tree[vy][0] != {}:
                new_tree = new_tree[vy]
            else:
                new_tree = resettrie
        elif vy not in new_tree:
            new_tree = [{}]
        else:
            new_tree = new_tree[vy]
        return new_tree

    def join(self, source_encodings: torch.Tensor, source_lengths: torch.Tensor, target_encodings: torch.Tensor, target_lengths: torch.Tensor, hptr: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies joint network to source and target encodings.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.
        A: TCPGen attention dimension

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
            hptr (torch.Tensor): deep biasing vector with shape `(B, T, U, A)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    joint network second last layer output, with shape `(B, T, U, D)`.
        """
        output, source_lengths, target_lengths, jointer_activation = self.joiner(source_encodings=source_encodings, source_lengths=source_lengths, target_encodings=target_encodings, target_lengths=target_lengths, hptr=hptr)
        return (output, source_lengths, jointer_activation)