import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _bind_event_handlers(self):
    self.top.bind(CORPUS_LOADED_EVENT, self.handle_corpus_loaded)
    self.top.bind(SEARCH_TERMINATED_EVENT, self.handle_search_terminated)
    self.top.bind(SEARCH_ERROR_EVENT, self.handle_search_error)
    self.top.bind(ERROR_LOADING_CORPUS_EVENT, self.handle_error_loading_corpus)