import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
@dataclass
class RNNTBundle:
    """Dataclass that bundles components for performing automatic speech recognition (ASR, speech-to-text)
    inference with an RNN-T model.

    More specifically, the class provides methods that produce the featurization pipeline,
    decoder wrapping the specified RNN-T model, and output token post-processor that together
    constitute a complete end-to-end ASR inference pipeline that produces a text sequence
    given a raw waveform.

    It can support non-streaming (full-context) inference as well as streaming inference.

    Users should not directly instantiate objects of this class; rather, users should use the
    instances (representing pre-trained models) that exist within the module,
    e.g. :data:`torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH`.

    Example
        >>> import torchaudio
        >>> from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
        >>> import torch
        >>>
        >>> # Non-streaming inference.
        >>> # Build feature extractor, decoder with RNN-T model, and token processor.
        >>> feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
        100%|███████████████████████████████| 3.81k/3.81k [00:00<00:00, 4.22MB/s]
        >>> decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/emformer_rnnt_base_librispeech.pt"
        100%|███████████████████████████████| 293M/293M [00:07<00:00, 42.1MB/s]
        >>> token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()
        100%|███████████████████████████████| 295k/295k [00:00<00:00, 25.4MB/s]
        >>>
        >>> # Instantiate LibriSpeech dataset; retrieve waveform for first sample.
        >>> dataset = torchaudio.datasets.LIBRISPEECH("/home/librispeech", url="test-clean")
        >>> waveform = next(iter(dataset))[0].squeeze()
        >>>
        >>> with torch.no_grad():
        >>>     # Produce mel-scale spectrogram features.
        >>>     features, length = feature_extractor(waveform)
        >>>
        >>>     # Generate top-10 hypotheses.
        >>>     hypotheses = decoder(features, length, 10)
        >>>
        >>> # For top hypothesis, convert predicted tokens to text.
        >>> text = token_processor(hypotheses[0][0])
        >>> print(text)
        he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to [...]
        >>>
        >>>
        >>> # Streaming inference.
        >>> hop_length = EMFORMER_RNNT_BASE_LIBRISPEECH.hop_length
        >>> num_samples_segment = EMFORMER_RNNT_BASE_LIBRISPEECH.segment_length * hop_length
        >>> num_samples_segment_right_context = (
        >>>     num_samples_segment + EMFORMER_RNNT_BASE_LIBRISPEECH.right_context_length * hop_length
        >>> )
        >>>
        >>> # Build streaming inference feature extractor.
        >>> streaming_feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_streaming_feature_extractor()
        >>>
        >>> # Process same waveform as before, this time sequentially across overlapping segments
        >>> # to simulate streaming inference. Note the usage of ``streaming_feature_extractor`` and ``decoder.infer``.
        >>> state, hypothesis = None, None
        >>> for idx in range(0, len(waveform), num_samples_segment):
        >>>     segment = waveform[idx: idx + num_samples_segment_right_context]
        >>>     segment = torch.nn.functional.pad(segment, (0, num_samples_segment_right_context - len(segment)))
        >>>     with torch.no_grad():
        >>>         features, length = streaming_feature_extractor(segment)
        >>>         hypotheses, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
        >>>     hypothesis = hypotheses[0]
        >>>     transcript = token_processor(hypothesis[0])
        >>>     if transcript:
        >>>         print(transcript, end=" ", flush=True)
        he hoped there would be stew for dinner turn ips and car rots and bru 'd oes and fat mut ton pieces to [...]
    """

    class FeatureExtractor(_FeatureExtractor):
        """Interface of the feature extraction part of RNN-T pipeline"""

    class TokenProcessor(_TokenProcessor):
        """Interface of the token processor part of RNN-T pipeline"""
    _rnnt_path: str
    _rnnt_factory_func: Callable[[], RNNT]
    _global_stats_path: str
    _sp_model_path: str
    _right_padding: int
    _blank: int
    _sample_rate: int
    _n_fft: int
    _n_mels: int
    _hop_length: int
    _segment_length: int
    _right_context_length: int

    def _get_model(self) -> RNNT:
        model = self._rnnt_factory_func()
        path = torchaudio.utils.download_asset(self._rnnt_path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @property
    def sample_rate(self) -> int:
        """Sample rate (in cycles per second) of input waveforms.

        :type: int
        """
        return self._sample_rate

    @property
    def n_fft(self) -> int:
        """Size of FFT window to use.

        :type: int
        """
        return self._n_fft

    @property
    def n_mels(self) -> int:
        """Number of mel spectrogram features to extract from input waveforms.

        :type: int
        """
        return self._n_mels

    @property
    def hop_length(self) -> int:
        """Number of samples between successive frames in input expected by model.

        :type: int
        """
        return self._hop_length

    @property
    def segment_length(self) -> int:
        """Number of frames in segment in input expected by model.

        :type: int
        """
        return self._segment_length

    @property
    def right_context_length(self) -> int:
        """Number of frames in right contextual block in input expected by model.

        :type: int
        """
        return self._right_context_length

    def get_decoder(self) -> RNNTBeamSearch:
        """Constructs RNN-T decoder.

        Returns:
            RNNTBeamSearch
        """
        model = self._get_model()
        return RNNTBeamSearch(model, self._blank)

    def get_feature_extractor(self) -> FeatureExtractor:
        """Constructs feature extractor for non-streaming (full-context) ASR.

        Returns:
            FeatureExtractor
        """
        local_path = torchaudio.utils.download_asset(self._global_stats_path)
        return _ModuleFeatureExtractor(torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length), _FunctionalModule(lambda x: x.transpose(1, 0)), _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)), _GlobalStatsNormalization(local_path), _FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 0, 0, self._right_padding)))))

    def get_streaming_feature_extractor(self) -> FeatureExtractor:
        """Constructs feature extractor for streaming (simultaneous) ASR.

        Returns:
            FeatureExtractor
        """
        local_path = torchaudio.utils.download_asset(self._global_stats_path)
        return _ModuleFeatureExtractor(torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length), _FunctionalModule(lambda x: x.transpose(1, 0)), _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)), _GlobalStatsNormalization(local_path)))

    def get_token_processor(self) -> TokenProcessor:
        """Constructs token processor.

        Returns:
            TokenProcessor
        """
        local_path = torchaudio.utils.download_asset(self._sp_model_path)
        return _SentencePieceTokenProcessor(local_path)