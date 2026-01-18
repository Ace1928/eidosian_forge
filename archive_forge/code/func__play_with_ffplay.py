import subprocess
from tempfile import NamedTemporaryFile
from .utils import get_player_name, make_chunks
def _play_with_ffplay(seg):
    PLAYER = get_player_name()
    with NamedTemporaryFile('w+b', suffix='.wav') as f:
        seg.export(f.name, 'wav')
        subprocess.call([PLAYER, '-nodisp', '-autoexit', '-hide_banner', f.name])