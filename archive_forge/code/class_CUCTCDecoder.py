from __future__ import annotations
import math
from typing import List, NamedTuple, Union
import torch
import torchaudio
import torchaudio.lib.pybind11_prefixctc as cuctc
class CUCTCDecoder:
    """CUDA CTC beam search decoder.

    .. devices:: CUDA

    Note:
        To build the decoder, please use the factory function :func:`cuda_ctc_decoder`.
    """

    def __init__(self, vocab_list: List[str], blank_id: int=0, beam_size: int=10, nbest: int=1, blank_skip_threshold: float=_DEFAULT_BLANK_SKIP_THREASHOLD, cuda_stream: torch.cuda.streams.Stream=None):
        """
        Args:
            blank_id (int): token id corresopnding to blank, only support 0 for now. (Default: 0)
            vocab_list (List[str]): list of vocabulary tokens
            beam_size (int, optional): max number of hypos to hold after each decode step (Default: 10)
            nbest (int): number of best decodings to return
            blank_skip_threshold (float):
                skip frames if log_prob(blank) > log(blank_skip_threshold), to speed up decoding.
                (Default: 0.95).
            cuda_stream (torch.cuda.streams.Stream): using assigned cuda stream (Default: using default stream)

        """
        if cuda_stream:
            if not isinstance(cuda_stream, torch.cuda.streams.Stream):
                raise AssertionError('cuda_stream must be torch.cuda.streams.Stream')
        cuda_stream_ = cuda_stream.cuda_stream if cuda_stream else torch.cuda.current_stream().cuda_stream
        self.internal_data = cuctc.prefixCTC_alloc(cuda_stream_)
        self.memory = torch.empty(0, dtype=torch.int8, device=torch.device('cuda'))
        if blank_id != 0:
            raise AssertionError('blank_id must be 0')
        self.blank_id = blank_id
        self.vocab_list = vocab_list
        self.space_id = 0
        self.nbest = nbest
        if not (blank_skip_threshold >= 0 and blank_skip_threshold <= 1):
            raise AssertionError('blank_skip_threshold must be between 0 and 1')
        self.blank_skip_threshold = math.log(blank_skip_threshold)
        self.beam_size = min(beam_size, len(vocab_list))

    def __del__(self):
        if cuctc is not None:
            cuctc.prefixCTC_free(self.internal_data)

    def __call__(self, log_prob: torch.Tensor, encoder_out_lens: torch.Tensor):
        """
        Args:
            log_prob (torch.FloatTensor): GPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; log_softmax(output of acoustic model).
            lengths (dtype torch.int32): GPU tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch.

        Returns:
            List[List[CUCTCHypothesis]]:
                List of sorted best hypotheses for each audio sequence in the batch.
        """
        if not encoder_out_lens.dtype == torch.int32:
            raise AssertionError('encoder_out_lens must be torch.int32')
        if not log_prob.dtype == torch.float32:
            raise AssertionError('log_prob must be torch.float32')
        if not (log_prob.is_cuda and encoder_out_lens.is_cuda):
            raise AssertionError('inputs must be cuda tensors')
        if not (log_prob.is_contiguous() and encoder_out_lens.is_contiguous()):
            raise AssertionError('input tensors must be contiguous')
        required_size, score_hyps = cuctc.ctc_beam_search_decoder_batch_gpu_v2(self.internal_data, self.memory.data_ptr(), self.memory.size(0), log_prob.data_ptr(), encoder_out_lens.data_ptr(), log_prob.size(), log_prob.stride(), self.beam_size, self.blank_id, self.space_id, self.blank_skip_threshold)
        if required_size > 0:
            self.memory = torch.empty(required_size, dtype=torch.int8, device=log_prob.device).contiguous()
            _, score_hyps = cuctc.ctc_beam_search_decoder_batch_gpu_v2(self.internal_data, self.memory.data_ptr(), self.memory.size(0), log_prob.data_ptr(), encoder_out_lens.data_ptr(), log_prob.size(), log_prob.stride(), self.beam_size, self.blank_id, self.space_id, self.blank_skip_threshold)
        batch_size = len(score_hyps)
        hypos = []
        for i in range(batch_size):
            hypos.append([CUCTCHypothesis(tokens=score_hyps[i][j][1], words=[self.vocab_list[word_id] for word_id in score_hyps[i][j][1]], score=score_hyps[i][j][0]) for j in range(self.nbest)])
        return hypos