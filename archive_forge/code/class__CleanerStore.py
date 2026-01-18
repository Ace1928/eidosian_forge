import threading
from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown
from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging
class _CleanerStore(threading.local):

    def __init__(self):
        self.cleaner = Cleaner(tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRIBUTES)