from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from torch.nn import Module
from . import aligner, utils
@dataclass
class Wav2Vec2ASRBundle(Wav2Vec2Bundle):
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - ASR
        >>> import torchaudio
        >>>
        >>> bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 1.18G/1.18G [00:17<00:00, 73.8MB/s]
        >>>
        >>> # Check the corresponding labels of the output.
        >>> labels = bundle.get_labels()
        >>> print(labels)
        ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Infer the label probability distribution
        >>> emissions, _ = model(waveform)
        >>>
        >>> # Pass emission to decoder
        >>> # `ctc_decode` is for illustration purpose only
        >>> transcripts = ctc_decode(emissions, labels)
    """
    _labels: Tuple[str, ...]
    _remove_aux_axis: Tuple[int, ...] = (1, 2, 3)

    def get_labels(self, *, blank: str='-') -> Tuple[str, ...]:
        """The output class labels.

        The first is blank token, and it is customizable.

        Args:
            blank (str, optional): Blank token. (default: ``'-'``)

        Returns:
            Tuple[str, ...]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> from torchaudio.pipelines import HUBERT_ASR_LARGE as bundle
            >>> bundle.get_labels()
            ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        """
        return (blank, *self._labels)

    def _get_state_dict(self, dl_kwargs):
        return utils._get_state_dict(self._path, dl_kwargs, self._remove_aux_axis)