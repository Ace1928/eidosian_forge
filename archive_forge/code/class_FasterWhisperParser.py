import logging
import os
import time
from typing import Dict, Iterator, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utils.openai import is_openai_v1
class FasterWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files with faster-whisper.

    faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2,
    which is up to 4 times faster than openai/whisper for the same accuracy while using
    less memory. The efficiency can be further improved with 8-bit quantization on both
    CPU and GPU.

    It can automatically detect the following 14 languages and transcribe the text
    into their respective languages: en, zh, fr, de, ja, ko, ru, es, th, it, pt, vi,
    ar, tr.

    The gitbub repository for faster-whisper is :
    https://github.com/SYSTRAN/faster-whisper

    Example: Load a YouTube video and transcribe the video speech into a document.
        .. code-block:: python

            from langchain.document_loaders.generic import GenericLoader
            from langchain_community.document_loaders.parsers.audio
                import FasterWhisperParser
            from langchain.document_loaders.blob_loaders.youtube_audio
                import YoutubeAudioLoader


            url="https://www.youtube.com/watch?v=your_video"
            save_dir="your_dir/"
            loader = GenericLoader(
                YoutubeAudioLoader([url],save_dir),
                FasterWhisperParser()
            )
            docs = loader.load()

    """

    def __init__(self, *, device: Optional[str]='cuda', model_size: Optional[str]=None):
        """Initialize the parser.

        Args:
            device: It can be "cuda" or "cpu" based on the available device.
            model_size: There are four model sizes to choose from: "base", "small",
                        "medium", and "large-v3", based on the available GPU memory.
        """
        try:
            import torch
        except ImportError:
            raise ImportError('torch package not found, please install it with `pip install torch`')
        if device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.model_size = 'base'
        else:
            mem = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2
            if mem < 1000:
                self.model_size = 'base'
            elif mem < 3000:
                self.model_size = 'small'
            elif mem < 5000:
                self.model_size = 'medium'
            else:
                self.model_size = 'large-v3'
        if model_size is not None:
            if model_size in ['base', 'small', 'medium', 'large-v3']:
                self.model_size = model_size

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import io
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError('pydub package not found, please install it with `pip install pydub`')
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError('faster_whisper package not found, please install it with `pip install faster-whisper`')
        if isinstance(blob.data, bytes):
            audio = AudioSegment.from_file(io.BytesIO(blob.data))
        elif blob.data is None and blob.path:
            audio = AudioSegment.from_file(blob.path)
        else:
            raise ValueError('Unable to get audio from blob')
        file_obj = io.BytesIO(audio.export(format='mp3').read())
        model = WhisperModel(self.model_size, device=self.device, compute_type='float16')
        segments, info = model.transcribe(file_obj, beam_size=5)
        for segment in segments:
            yield Document(page_content=segment.text, metadata={'source': blob.source, 'timestamps': '[%.2fs -> %.2fs]' % (segment.start, segment.end), 'language': info.language, 'probability': '%d%%' % round(info.language_probability * 100), **blob.metadata})