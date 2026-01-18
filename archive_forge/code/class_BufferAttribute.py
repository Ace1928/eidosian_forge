from ipywidgets import register
from .._base.Three import ThreeWidget
from .BufferAttribute_autogen import BufferAttribute as BaseBufferAttribute
@register
class BufferAttribute(BaseBufferAttribute):
    _previewable = False

    def __init__(self, array=None, normalized=True, **kwargs):
        if array is not None:
            kwargs['array'] = array
        kwargs['normalized'] = normalized
        super(BaseBufferAttribute, self).__init__(**kwargs)