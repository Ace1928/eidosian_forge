import subprocess
from tempfile import NamedTemporaryFile
from .utils import get_player_name, make_chunks
def _play_with_simpleaudio(seg):
    import simpleaudio
    return simpleaudio.play_buffer(seg.raw_data, num_channels=seg.channels, bytes_per_sample=seg.sample_width, sample_rate=seg.frame_rate)