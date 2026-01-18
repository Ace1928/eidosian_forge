import logging
import os
import time
from typing import Dict, Iterator, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utils.openai import is_openai_v1
class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.

    Audio transcription is with OpenAI Whisper model.

    Args:
        api_key: OpenAI API key
        chunk_duration_threshold: minimum duration of a chunk in seconds
            NOTE: According to the OpenAI API, the chunk duration should be at least 0.1
            seconds. If the chunk duration is less or equal than the threshold,
            it will be skipped.
    """

    def __init__(self, api_key: Optional[str]=None, *, chunk_duration_threshold: float=0.1, base_url: Optional[str]=None):
        self.api_key = api_key
        self.chunk_duration_threshold = chunk_duration_threshold
        self.base_url = base_url if base_url is not None else os.environ.get('OPENAI_API_BASE')

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import io
        try:
            import openai
        except ImportError:
            raise ImportError('openai package not found, please install it with `pip install openai`')
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError('pydub package not found, please install it with `pip install pydub`')
        if is_openai_v1():
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            if self.api_key:
                openai.api_key = self.api_key
            if self.base_url:
                openai.base_url = self.base_url
        audio = AudioSegment.from_file(blob.path)
        chunk_duration = 20
        chunk_duration_ms = chunk_duration * 60 * 1000
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            chunk = audio[i:i + chunk_duration_ms]
            if chunk.duration_seconds <= self.chunk_duration_threshold:
                continue
            file_obj = io.BytesIO(chunk.export(format='mp3').read())
            if blob.source is not None:
                file_obj.name = blob.source + f'_part_{split_number}.mp3'
            else:
                file_obj.name = f'part_{split_number}.mp3'
            print(f'Transcribing part {split_number + 1}!')
            attempts = 0
            while attempts < 3:
                try:
                    if is_openai_v1():
                        transcript = client.audio.transcriptions.create(model='whisper-1', file=file_obj)
                    else:
                        transcript = openai.Audio.transcribe('whisper-1', file_obj)
                    break
                except Exception as e:
                    attempts += 1
                    print(f'Attempt {attempts} failed. Exception: {str(e)}')
                    time.sleep(5)
            else:
                print('Failed to transcribe after 3 attempts.')
                continue
            yield Document(page_content=transcript.text, metadata={'source': blob.source, 'chunk': split_number})