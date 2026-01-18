import atexit
import tempfile
def _cleanup_media_tmp_dir() -> None:
    atexit.register(MEDIA_TMP.cleanup)