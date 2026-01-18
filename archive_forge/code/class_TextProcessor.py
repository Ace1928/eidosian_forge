from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
class TextProcessor(_TextProcessor):
    """Interface of the text processing part of Tacotron2TTS pipeline

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_text_processor` for the usage.
        """