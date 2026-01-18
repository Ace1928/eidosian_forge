import logging
import os
import time
from typing import Dict, Iterator, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utils.openai import is_openai_v1
class OpenAIWhisperParserLocal(BaseBlobParser):
    """Transcribe and parse audio files with OpenAI Whisper model.

    Audio transcription with OpenAI Whisper model locally from transformers.

    Parameters:
    device - device to use
        NOTE: By default uses the gpu if available,
        if you want to use cpu, please set device = "cpu"
    lang_model - whisper model to use, for example "openai/whisper-medium"
    forced_decoder_ids - id states for decoder in multilanguage model,
        usage example:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="french",
          task="transcribe")
        forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="french",
        task="translate")



    """

    def __init__(self, device: str='0', lang_model: Optional[str]=None, batch_size: int=8, chunk_length: int=30, forced_decoder_ids: Optional[Tuple[Dict]]=None):
        """Initialize the parser.

        Args:
            device: device to use.
            lang_model: whisper model to use, for example "openai/whisper-medium".
              Defaults to None.
            forced_decoder_ids: id states for decoder in a multilanguage model.
              Defaults to None.
            batch_size: batch size used for decoding
              Defaults to 8.
            chunk_length: chunk length used during inference.
              Defaults to 30s.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError('transformers package not found, please install it with `pip install transformers`')
        try:
            import torch
        except ImportError:
            raise ImportError('torch package not found, please install it with `pip install torch`')
        if device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            default_model = 'openai/whisper-base'
            self.lang_model = lang_model if lang_model else default_model
        else:
            mem = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2
            if mem < 5000:
                rec_model = 'openai/whisper-base'
            elif mem < 7000:
                rec_model = 'openai/whisper-small'
            elif mem < 12000:
                rec_model = 'openai/whisper-medium'
            else:
                rec_model = 'openai/whisper-large'
            self.lang_model = lang_model if lang_model else rec_model
        print('Using the following model: ', self.lang_model)
        self.batch_size = batch_size
        self.pipe = pipeline('automatic-speech-recognition', model=self.lang_model, chunk_length_s=chunk_length, device=self.device)
        if forced_decoder_ids is not None:
            try:
                self.pipe.model.config.forced_decoder_ids = forced_decoder_ids
            except Exception as exception_text:
                logger.info(f'Unable to set forced_decoder_ids parameter for whisper modelText of exception: {exception_text}Therefore whisper model will use default mode for decoder')

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import io
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError('pydub package not found, please install it with `pip install pydub`')
        try:
            import librosa
        except ImportError:
            raise ImportError('librosa package not found, please install it with `pip install librosa`')
        audio = AudioSegment.from_file(blob.path)
        file_obj = io.BytesIO(audio.export(format='mp3').read())
        print(f'Transcribing part {blob.path}!')
        y, sr = librosa.load(file_obj, sr=16000)
        prediction = self.pipe(y.copy(), batch_size=self.batch_size)['text']
        yield Document(page_content=prediction, metadata={'source': blob.source})