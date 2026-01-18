import gc
import re
import nltk
def _unload(self):
    lazy_reader = LazyCorpusLoader(name, reader_cls, *args, **kwargs)
    self.__dict__ = lazy_reader.__dict__
    self.__class__ = lazy_reader.__class__
    gc.collect()