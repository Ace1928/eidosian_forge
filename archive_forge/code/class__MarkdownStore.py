import threading
from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown
from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging
class _MarkdownStore(threading.local):

    def __init__(self):
        self.markdown = markdown.Markdown(extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code'])