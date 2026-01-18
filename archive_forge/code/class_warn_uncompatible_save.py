import warnings
from ...utils.import_utils import check_if_transformers_greater
from .decoder_models import (
from .encoder_models import (
class warn_uncompatible_save(object):

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        warnings.warn('You are calling `save_pretrained` to a `BetterTransformer` converted model you may likely encounter unexepected behaviors. ', UserWarning)
        return self.callback(*args, **kwargs)