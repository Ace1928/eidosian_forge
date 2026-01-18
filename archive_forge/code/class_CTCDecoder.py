from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
class CTCDecoder:
    """CTC beam search decoder from *Flashlight* :cite:`kahn2022flashlight`.

    .. devices:: CPU

    Note:
        To build the decoder, please use the factory function :func:`ctc_decoder`.
    """

    def __init__(self, nbest: int, lexicon: Optional[Dict], word_dict: _Dictionary, tokens_dict: _Dictionary, lm: CTCDecoderLM, decoder_options: Union[_LexiconDecoderOptions, _LexiconFreeDecoderOptions], blank_token: str, sil_token: str, unk_word: str) -> None:
        """
        Args:
            nbest (int): number of best decodings to return
            lexicon (Dict or None): lexicon mapping of words to spellings, or None for lexicon-free decoder
            word_dict (_Dictionary): dictionary of words
            tokens_dict (_Dictionary): dictionary of tokens
            lm (CTCDecoderLM): language model. If using a lexicon, only word level LMs are currently supported
            decoder_options (_LexiconDecoderOptions or _LexiconFreeDecoderOptions):
                parameters used for beam search decoding
            blank_token (str): token corresopnding to blank
            sil_token (str): token corresponding to silence
            unk_word (str): word corresponding to unknown
        """
        self.nbest = nbest
        self.word_dict = word_dict
        self.tokens_dict = tokens_dict
        self.blank = self.tokens_dict.get_index(blank_token)
        silence = self.tokens_dict.get_index(sil_token)
        transitions = []
        if lexicon:
            trie = _construct_trie(tokens_dict, word_dict, lexicon, lm, silence)
            unk_word = word_dict.get_index(unk_word)
            token_lm = False
            self.decoder = _LexiconDecoder(decoder_options, trie, lm, silence, self.blank, unk_word, transitions, token_lm)
        else:
            self.decoder = _LexiconFreeDecoder(decoder_options, lm, silence, self.blank, transitions)
        self.lm = lm

    def _get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def _get_timesteps(self, idxs: torch.IntTensor) -> torch.IntTensor:
        """Returns frame numbers corresponding to non-blank tokens."""
        timesteps = []
        for i, idx in enumerate(idxs):
            if idx == self.blank:
                continue
            if i == 0 or idx != idxs[i - 1]:
                timesteps.append(i)
        return torch.IntTensor(timesteps)

    def decode_begin(self):
        """Initialize the internal state of the decoder.

        See :py:meth:`decode_step` for the usage.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.
        """
        self.decoder.decode_begin()

    def decode_end(self):
        """Finalize the internal state of the decoder.

        See :py:meth:`decode_step` for the usage.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.
        """
        self.decoder.decode_end()

    def decode_step(self, emissions: torch.FloatTensor):
        """Perform incremental decoding on top of the curent internal state.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.

        Args:
            emissions (torch.FloatTensor): CPU tensor of shape `(frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.

        Example:
            >>> decoder = torchaudio.models.decoder.ctc_decoder(...)
            >>> decoder.decode_begin()
            >>> decoder.decode_step(emission1)
            >>> decoder.decode_step(emission2)
            >>> decoder.decode_end()
            >>> result = decoder.get_final_hypothesis()
        """
        if emissions.dtype != torch.float32:
            raise ValueError('emissions must be float32.')
        if not emissions.is_cpu:
            raise RuntimeError('emissions must be a CPU tensor.')
        if not emissions.is_contiguous():
            raise RuntimeError('emissions must be contiguous.')
        if emissions.ndim != 2:
            raise RuntimeError(f'emissions must be 2D. Found {emissions.shape}')
        T, N = emissions.size()
        self.decoder.decode_step(emissions.data_ptr(), T, N)

    def _to_hypo(self, results) -> List[CTCHypothesis]:
        return [CTCHypothesis(tokens=self._get_tokens(result.tokens), words=[self.word_dict.get_entry(x) for x in result.words if x >= 0], score=result.score, timesteps=self._get_timesteps(result.tokens)) for result in results]

    def get_final_hypothesis(self) -> List[CTCHypothesis]:
        """Get the final hypothesis

        Returns:
            List[CTCHypothesis]:
                List of sorted best hypotheses.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.
        """
        results = self.decoder.get_all_final_hypothesis()
        return self._to_hypo(results[:self.nbest])

    def __call__(self, emissions: torch.FloatTensor, lengths: Optional[torch.Tensor]=None) -> List[List[CTCHypothesis]]:
        """
        Performs batched offline decoding.

        .. note::

           This method performs offline decoding in one go. To perform incremental decoding,
           please refer to :py:meth:`decode_step`.

        Args:
            emissions (torch.FloatTensor): CPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.
            lengths (Tensor or None, optional): CPU tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch.

        Returns:
            List[List[CTCHypothesis]]:
                List of sorted best hypotheses for each audio sequence in the batch.
        """
        if emissions.dtype != torch.float32:
            raise ValueError('emissions must be float32.')
        if not emissions.is_cpu:
            raise RuntimeError('emissions must be a CPU tensor.')
        if not emissions.is_contiguous():
            raise RuntimeError('emissions must be contiguous.')
        if emissions.ndim != 3:
            raise RuntimeError(f'emissions must be 3D. Found {emissions.shape}')
        if lengths is not None and (not lengths.is_cpu):
            raise RuntimeError('lengths must be a CPU tensor.')
        B, T, N = emissions.size()
        if lengths is None:
            lengths = torch.full((B,), T)
        float_bytes = 4
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + float_bytes * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, lengths[b], N)
            hypos.append(self._to_hypo(results[:self.nbest]))
        return hypos

    def idxs_to_tokens(self, idxs: torch.LongTensor) -> List:
        """
        Map raw token IDs into corresponding tokens

        Args:
            idxs (LongTensor): raw token IDs generated from decoder

        Returns:
            List: tokens corresponding to the input IDs
        """
        return [self.tokens_dict.get_entry(idx.item()) for idx in idxs]