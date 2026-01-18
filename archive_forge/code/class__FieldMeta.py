import types
import weakref
import six
from apitools.base.protorpclite import util
class _FieldMeta(type):

    def __init__(cls, name, bases, dct):
        getattr(cls, '_Field__variant_to_type').update(((variant, cls) for variant in dct.get('VARIANTS', [])))
        type.__init__(cls, name, bases, dct)