from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
class RNNT(torch.nn.Module):
    """torchaudio.models.RNNT()

    Recurrent neural network transducer (RNN-T) model.

    Note:
        To build the model, please use one of the factory functions.

    See Also:
        :class:`torchaudio.pipelines.RNNTBundle`: ASR pipeline with pre-trained models.

    Args:
        transcriber (torch.nn.Module): transcription network.
        predictor (torch.nn.Module): prediction network.
        joiner (torch.nn.Module): joint network.
    """

    def __init__(self, transcriber: _Transcriber, predictor: _Predictor, joiner: _Joiner) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner

    def forward(self, sources: torch.Tensor, source_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, predictor_state: Optional[List[List[torch.Tensor]]]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
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
        """
        source_encodings, source_lengths = self.transcriber(input=sources, lengths=source_lengths)
        target_encodings, target_lengths, predictor_state = self.predictor(input=targets, lengths=target_lengths, state=predictor_state)
        output, source_lengths, target_lengths = self.joiner(source_encodings=source_encodings, source_lengths=source_lengths, target_encodings=target_encodings, target_lengths=target_lengths)
        return (output, source_lengths, target_lengths, predictor_state)

    @torch.jit.export
    def transcribe_streaming(self, sources: torch.Tensor, source_lengths: torch.Tensor, state: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Applies transcription network to sources in streaming mode.

        B: batch size;
        T: maximum source sequence segment length in batch;
        D: feature dimension of each source sequence frame.

        Args:
            sources (torch.Tensor): source frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing transcription network internal state generated in preceding invocation
                of ``transcribe_streaming``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing transcription network internal state generated in current invocation
                    of ``transcribe_streaming``.
        """
        return self.transcriber.infer(sources, source_lengths, state)

    @torch.jit.export
    def transcribe(self, sources: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies transcription network to sources in non-streaming mode.

        B: batch size;
        T: maximum source sequence length in batch;
        D: feature dimension of each source sequence frame.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        return self.transcriber(sources, source_lengths)

    @torch.jit.export
    def predict(self, targets: torch.Tensor, target_lengths: torch.Tensor, state: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Applies prediction network to targets.

        B: batch size;
        U: maximum target sequence length in batch;
        D: feature dimension of each target sequence frame.

        Args:
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``predict``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with shape `(B, U, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``predict``.
        """
        return self.predictor(input=targets, lengths=target_lengths, state=state)

    @torch.jit.export
    def join(self, source_encodings: torch.Tensor, source_lengths: torch.Tensor, target_encodings: torch.Tensor, target_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies joint network to source and target encodings.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        output, source_lengths, target_lengths = self.joiner(source_encodings=source_encodings, source_lengths=source_lengths, target_encodings=target_encodings, target_lengths=target_lengths)
        return (output, source_lengths, target_lengths)