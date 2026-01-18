import base64
import os
from contextlib import contextmanager
from io import BytesIO
from itertools import chain
from tempfile import NamedTemporaryFile
import matplotlib as mpl
import param
from matplotlib import pyplot as plt
from param.parameterized import bothmethod
from ...core import HoloMap
from ...core.options import Store
from ..renderer import HTML_TAGS, MIME_TYPES, Renderer
from .util import get_old_rcparams, get_tight_bbox
def _anim_data(self, anim, fmt):
    """
        Render a matplotlib animation object and return the corresponding data.
        """
    writer, _, anim_kwargs, extra_args = ANIMATION_OPTS[fmt]
    if extra_args != []:
        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)
    if self.fps is not None:
        anim_kwargs['fps'] = max([int(self.fps), 1])
    if self.dpi is not None:
        anim_kwargs['dpi'] = self.dpi
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as f:
            anim.save(f.name, writer=writer, **anim_kwargs)
            video = f.read()
        f.close()
        os.remove(f.name)
    return video