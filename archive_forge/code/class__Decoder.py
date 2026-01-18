import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class _Decoder(nn.Module):
    """Decoder with Attention model.

    Args:
        n_mels (int): number of mel bins
        n_frames_per_step (int): number of frames processed per step, only 1 is supported
        encoder_embedding_dim (int): the number of embedding dimensions in the encoder.
        decoder_rnn_dim (int): number of units in decoder LSTM
        decoder_max_step (int): maximum number of output mel spectrograms
        decoder_dropout (float): dropout probability for decoder LSTM
        decoder_early_stopping (bool): stop decoding when all samples are finished
        attention_rnn_dim (int): number of units in attention LSTM
        attention_hidden_dim (int): dimension of attention hidden representation
        attention_location_n_filter (int): number of filters for attention model
        attention_location_kernel_size (int): kernel size for attention model
        attention_dropout (float): dropout probability for attention LSTM
        prenet_dim (int): number of ReLU units in prenet layers
        gate_threshold (float): probability threshold for stop token
    """

    def __init__(self, n_mels: int, n_frames_per_step: int, encoder_embedding_dim: int, decoder_rnn_dim: int, decoder_max_step: int, decoder_dropout: float, decoder_early_stopping: bool, attention_rnn_dim: int, attention_hidden_dim: int, attention_location_n_filter: int, attention_location_kernel_size: int, attention_dropout: float, prenet_dim: int, gate_threshold: float) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.decoder_max_step = decoder_max_step
        self.gate_threshold = gate_threshold
        self.attention_dropout = attention_dropout
        self.decoder_dropout = decoder_dropout
        self.decoder_early_stopping = decoder_early_stopping
        self.prenet = _Prenet(n_mels * n_frames_per_step, [prenet_dim, prenet_dim])
        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_embedding_dim, attention_rnn_dim)
        self.attention_layer = _Attention(attention_rnn_dim, encoder_embedding_dim, attention_hidden_dim, attention_location_n_filter, attention_location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, True)
        self.linear_projection = _get_linear_layer(decoder_rnn_dim + encoder_embedding_dim, n_mels * n_frames_per_step)
        self.gate_layer = _get_linear_layer(decoder_rnn_dim + encoder_embedding_dim, 1, bias=True, w_init_gain='sigmoid')

    def _get_initial_frame(self, memory: Tensor) -> Tensor:
        """Gets all zeros frames to use as the first decoder input.

        Args:
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        Returns:
            decoder_input (Tensor): all zeros frames with shape
                (n_batch, max of ``text_lengths``, ``n_mels * n_frames_per_step``).
        """
        n_batch = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(n_batch, self.n_mels * self.n_frames_per_step, dtype=dtype, device=device)
        return decoder_input

    def _initialize_decoder_states(self, memory: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory.

        Args:
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        Returns:
            attention_hidden (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            attention_cell (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            decoder_hidden (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            decoder_cell (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
            attention_weights_cum (Tensor): Cumulated attention weights with shape (n_batch, max of ``text_lengths``).
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
            processed_memory (Tensor): Processed encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``attention_hidden_dim``).
        """
        n_batch = memory.size(0)
        max_time = memory.size(1)
        dtype = memory.dtype
        device = memory.device
        attention_hidden = torch.zeros(n_batch, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(n_batch, self.attention_rnn_dim, dtype=dtype, device=device)
        decoder_hidden = torch.zeros(n_batch, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(n_batch, self.decoder_rnn_dim, dtype=dtype, device=device)
        attention_weights = torch.zeros(n_batch, max_time, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(n_batch, max_time, dtype=dtype, device=device)
        attention_context = torch.zeros(n_batch, self.encoder_embedding_dim, dtype=dtype, device=device)
        processed_memory = self.attention_layer.memory_layer(memory)
        return (attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, processed_memory)

    def _parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        """Prepares decoder inputs.

        Args:
            decoder_inputs (Tensor): Inputs used for teacher-forced training, i.e. mel-specs,
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)

        Returns:
            inputs (Tensor): Processed decoder inputs with shape (max of ``mel_specgram_lengths``, n_batch, ``n_mels``).
        """
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def _parse_decoder_outputs(self, mel_specgram: Tensor, gate_outputs: Tensor, alignments: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepares decoder outputs for output

        Args:
            mel_specgram (Tensor): mel spectrogram with shape (max of ``mel_specgram_lengths``, n_batch, ``n_mels``)
            gate_outputs (Tensor): predicted stop token with shape (max of ``mel_specgram_lengths``, n_batch)
            alignments (Tensor): sequence of attention weights from the decoder
                with shape (max of ``mel_specgram_lengths``, n_batch, max of ``text_lengths``)

        Returns:
            mel_specgram (Tensor): mel spectrogram with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)
            gate_outputs (Tensor): predicted stop token with shape (n_batch, max of ``mel_specgram_lengths``)
            alignments (Tensor): sequence of attention weights from the decoder
                with shape (n_batch, max of ``mel_specgram_lengths``, max of ``text_lengths``)
        """
        alignments = alignments.transpose(0, 1).contiguous()
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        mel_specgram = mel_specgram.transpose(0, 1).contiguous()
        shape = (mel_specgram.shape[0], -1, self.n_mels)
        mel_specgram = mel_specgram.view(*shape)
        mel_specgram = mel_specgram.transpose(1, 2)
        return (mel_specgram, gate_outputs, alignments)

    def decode(self, decoder_input: Tensor, attention_hidden: Tensor, attention_cell: Tensor, decoder_hidden: Tensor, decoder_cell: Tensor, attention_weights: Tensor, attention_weights_cum: Tensor, attention_context: Tensor, memory: Tensor, processed_memory: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Decoder step using stored states, attention and memory

        Args:
            decoder_input (Tensor): Output of the Prenet with shape (n_batch, ``prenet_dim``).
            attention_hidden (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            attention_cell (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            decoder_hidden (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            decoder_cell (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
            attention_weights_cum (Tensor): Cumulated attention weights with shape (n_batch, max of ``text_lengths``).
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
            memory (Tensor): Encoder output with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``attention_hidden_dim``).
            mask (Tensor): Binary mask for padded data with shape (n_batch, current_num_frames).

        Returns:
            decoder_output: Predicted mel spectrogram for the current frame with shape (n_batch, ``n_mels``).
            gate_prediction (Tensor): Prediction of the stop token with shape (n_batch, ``1``).
            attention_hidden (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            attention_cell (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            decoder_hidden (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            decoder_cell (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
            attention_weights_cum (Tensor): Cumulated attention weights with shape (n_batch, max of ``text_lengths``).
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.attention_dropout, self.training)
        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(attention_hidden, memory, processed_memory, attention_weights_cat, mask)
        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.decoder_dropout, self.training)
        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return (decoder_output, gate_prediction, attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context)

    def forward(self, memory: Tensor, mel_specgram_truth: Tensor, memory_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Decoder forward pass for training.

        Args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            mel_specgram_truth (Tensor): Decoder ground-truth mel-specs for teacher forcing
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            memory_lengths (Tensor): Encoder output lengths for attention masking
                (the same as ``text_lengths``) with shape (n_batch, ).

        Returns:
            mel_specgram (Tensor): Predicted mel spectrogram
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            gate_outputs (Tensor): Predicted stop token for each timestep
                with shape (n_batch,  max of ``mel_specgram_lengths``).
            alignments (Tensor): Sequence of attention weights from the decoder
                with shape (n_batch,  max of ``mel_specgram_lengths``, max of ``text_lengths``).
        """
        decoder_input = self._get_initial_frame(memory).unsqueeze(0)
        decoder_inputs = self._parse_decoder_inputs(mel_specgram_truth)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        mask = _get_mask_from_lengths(memory_lengths)
        attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, processed_memory = self._initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = ([], [], [])
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context = self.decode(decoder_input, attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, memory, processed_memory, mask)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
        mel_specgram, gate_outputs, alignments = self._parse_decoder_outputs(torch.stack(mel_outputs), torch.stack(gate_outputs), torch.stack(alignments))
        return (mel_specgram, gate_outputs, alignments)

    def _get_go_frame(self, memory: Tensor) -> Tensor:
        """Gets all zeros frames to use as the first decoder input

        args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        returns:
            decoder_input (Tensor): All zeros frames with shape(n_batch, ``n_mels`` * ``n_frame_per_step``).
        """
        n_batch = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(n_batch, self.n_mels * self.n_frames_per_step, dtype=dtype, device=device)
        return decoder_input

    @torch.jit.export
    def infer(self, memory: Tensor, memory_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decoder inference

        Args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            memory_lengths (Tensor): Encoder output lengths for attention masking
                (the same as ``text_lengths``) with shape (n_batch, ).

        Returns:
            mel_specgram (Tensor): Predicted mel spectrogram
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            mel_specgram_lengths (Tensor): the length of the predicted mel spectrogram (n_batch, ))
            gate_outputs (Tensor): Predicted stop token for each timestep
                with shape (n_batch,  max of ``mel_specgram_lengths``).
            alignments (Tensor): Sequence of attention weights from the decoder
                with shape (n_batch,  max of ``mel_specgram_lengths``, max of ``text_lengths``).
        """
        batch_size, device = (memory.size(0), memory.device)
        decoder_input = self._get_go_frame(memory)
        mask = _get_mask_from_lengths(memory_lengths)
        attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, processed_memory = self._initialize_decoder_states(memory)
        mel_specgram_lengths = torch.zeros([batch_size], dtype=torch.int32, device=device)
        finished = torch.zeros([batch_size], dtype=torch.bool, device=device)
        mel_specgrams: List[Tensor] = []
        gate_outputs: List[Tensor] = []
        alignments: List[Tensor] = []
        for _ in range(self.decoder_max_step):
            decoder_input = self.prenet(decoder_input)
            mel_specgram, gate_output, attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context = self.decode(decoder_input, attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context, memory, processed_memory, mask)
            mel_specgrams.append(mel_specgram.unsqueeze(0))
            gate_outputs.append(gate_output.transpose(0, 1))
            alignments.append(attention_weights)
            mel_specgram_lengths[~finished] += 1
            finished |= torch.sigmoid(gate_output.squeeze(1)) > self.gate_threshold
            if self.decoder_early_stopping and torch.all(finished):
                break
            decoder_input = mel_specgram
        if len(mel_specgrams) == self.decoder_max_step:
            warnings.warn('Reached max decoder steps. The generated spectrogram might not cover the whole transcript.')
        mel_specgrams = torch.cat(mel_specgrams, dim=0)
        gate_outputs = torch.cat(gate_outputs, dim=0)
        alignments = torch.cat(alignments, dim=0)
        mel_specgrams, gate_outputs, alignments = self._parse_decoder_outputs(mel_specgrams, gate_outputs, alignments)
        return (mel_specgrams, mel_specgram_lengths, gate_outputs, alignments)