from .version import version as __version__
from . import monkeypatch
from .cool import *
from collections.abc import Mapping
def _getFrozendictJsonEncoder(BaseJsonEncoder=None):
    if BaseJsonEncoder is None:
        from json.encoder import JSONEncoder
        BaseJsonEncoder = JSONEncoder

    class FrozendictJsonEncoderInternal(BaseJsonEncoder):

        def default(self, obj):
            if isinstance(obj, frozendict):
                return dict(obj)
            return BaseJsonEncoder.default(self, obj)
    return FrozendictJsonEncoderInternal