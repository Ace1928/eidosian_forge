import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class Tacotron2(nn.Module):
    """Tacotron2 model from *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*
    :cite:`shen2018natural` based on the implementation from
    `Nvidia Deep Learning Examples <https://github.com/NVIDIA/DeepLearningExamples/>`_.

    See Also:
        * :class:`torchaudio.pipelines.Tacotron2TTSBundle`: TTS pipeline with pretrained model.

    Args:
        mask_padding (bool, optional): Use mask padding (Default: ``False``).
        n_mels (int, optional): Number of mel bins (Default: ``80``).
        n_symbol (int, optional): Number of symbols for the input text (Default: ``148``).
        n_frames_per_step (int, optional): Number of frames processed per step, only 1 is supported (Default: ``1``).
        symbol_embedding_dim (int, optional): Input embedding dimension (Default: ``512``).
        encoder_n_convolution (int, optional): Number of encoder convolutions (Default: ``3``).
        encoder_kernel_size (int, optional): Encoder kernel size (Default: ``5``).
        encoder_embedding_dim (int, optional): Encoder embedding dimension (Default: ``512``).
        decoder_rnn_dim (int, optional): Number of units in decoder LSTM (Default: ``1024``).
        decoder_max_step (int, optional): Maximum number of output mel spectrograms (Default: ``2000``).
        decoder_dropout (float, optional): Dropout probability for decoder LSTM (Default: ``0.1``).
        decoder_early_stopping (bool, optional): Continue decoding after all samples are finished (Default: ``True``).
        attention_rnn_dim (int, optional): Number of units in attention LSTM (Default: ``1024``).
        attention_hidden_dim (int, optional): Dimension of attention hidden representation (Default: ``128``).
        attention_location_n_filter (int, optional): Number of filters for attention model (Default: ``32``).
        attention_location_kernel_size (int, optional): Kernel size for attention model (Default: ``31``).
        attention_dropout (float, optional): Dropout probability for attention LSTM (Default: ``0.1``).
        prenet_dim (int, optional): Number of ReLU units in prenet layers (Default: ``256``).
        postnet_n_convolution (int, optional): Number of postnet convolutions (Default: ``5``).
        postnet_kernel_size (int, optional): Postnet kernel size (Default: ``5``).
        postnet_embedding_dim (int, optional): Postnet embedding dimension (Default: ``512``).
        gate_threshold (float, optional): Probability threshold for stop token (Default: ``0.5``).
    """

    def __init__(self, mask_padding: bool=False, n_mels: int=80, n_symbol: int=148, n_frames_per_step: int=1, symbol_embedding_dim: int=512, encoder_embedding_dim: int=512, encoder_n_convolution: int=3, encoder_kernel_size: int=5, decoder_rnn_dim: int=1024, decoder_max_step: int=2000, decoder_dropout: float=0.1, decoder_early_stopping: bool=True, attention_rnn_dim: int=1024, attention_hidden_dim: int=128, attention_location_n_filter: int=32, attention_location_kernel_size: int=31, attention_dropout: float=0.1, prenet_dim: int=256, postnet_n_convolution: int=5, postnet_kernel_size: int=5, postnet_embedding_dim: int=512, gate_threshold: float=0.5) -> None:
        super().__init__()
        self.mask_padding = mask_padding
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbol, symbol_embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.encoder = _Encoder(encoder_embedding_dim, encoder_n_convolution, encoder_kernel_size)
        self.decoder = _Decoder(n_mels, n_frames_per_step, encoder_embedding_dim, decoder_rnn_dim, decoder_max_step, decoder_dropout, decoder_early_stopping, attention_rnn_dim, attention_hidden_dim, attention_location_n_filter, attention_location_kernel_size, attention_dropout, prenet_dim, gate_threshold)
        self.postnet = _Postnet(n_mels, postnet_embedding_dim, postnet_kernel_size, postnet_n_convolution)

    def forward(self, tokens: Tensor, token_lengths: Tensor, mel_specgram: Tensor, mel_specgram_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pass the input through the Tacotron2 model. This is in teacher
        forcing mode, which is generally used for training.

        The input ``tokens`` should be padded with zeros to length max of ``token_lengths``.
        The input ``mel_specgram`` should be padded with zeros to length max of ``mel_specgram_lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of token_lengths)`.
            token_lengths (Tensor): The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
            mel_specgram (Tensor): The target mel spectrogram
                with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
            mel_specgram_lengths (Tensor): The length of each mel spectrogram with shape `(n_batch, )`.

        Returns:
            [Tensor, Tensor, Tensor, Tensor]:
                Tensor
                    Mel spectrogram before Postnet with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    Mel spectrogram after Postnet with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The output for stop token at each time step with shape `(n_batch, max of mel_specgram_lengths)`.
                Tensor
                    Sequence of attention weights from the decoder with
                    shape `(n_batch, max of mel_specgram_lengths, max of token_lengths)`.
        """
        embedded_inputs = self.embedding(tokens).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, token_lengths)
        mel_specgram, gate_outputs, alignments = self.decoder(encoder_outputs, mel_specgram, memory_lengths=token_lengths)
        mel_specgram_postnet = self.postnet(mel_specgram)
        mel_specgram_postnet = mel_specgram + mel_specgram_postnet
        if self.mask_padding:
            mask = _get_mask_from_lengths(mel_specgram_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            mel_specgram.masked_fill_(mask, 0.0)
            mel_specgram_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1000.0)
        return (mel_specgram, mel_specgram_postnet, gate_outputs, alignments)

    @torch.jit.export
    def infer(self, tokens: Tensor, lengths: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
        """Using Tacotron2 for inference. The input is a batch of encoded
        sentences (``tokens``) and its corresponding lengths (``lengths``). The
        output is the generated mel spectrograms, its corresponding lengths, and
        the attention weights from the decoder.

        The input `tokens` should be padded with zeros to length max of ``lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of lengths)`.
            lengths (Tensor or None, optional):
                The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
                If ``None``, it is assumed that the all the tokens are valid. Default: ``None``

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    The predicted mel spectrogram with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The length of the predicted mel spectrogram with shape `(n_batch, )`.
                Tensor
                    Sequence of attention weights from the decoder with shape
                    `(n_batch, max of mel_specgram_lengths, max of lengths)`.
        """
        n_batch, max_length = tokens.shape
        if lengths is None:
            lengths = torch.tensor([max_length]).expand(n_batch).to(tokens.device, tokens.dtype)
        assert lengths is not None
        embedded_inputs = self.embedding(tokens).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, lengths)
        mel_specgram, mel_specgram_lengths, _, alignments = self.decoder.infer(encoder_outputs, lengths)
        mel_outputs_postnet = self.postnet(mel_specgram)
        mel_outputs_postnet = mel_specgram + mel_outputs_postnet
        alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2)
        return (mel_outputs_postnet, mel_specgram_lengths, alignments)