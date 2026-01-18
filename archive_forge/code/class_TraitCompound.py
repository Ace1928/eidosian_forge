from importlib import import_module
import sys
from types import FunctionType, MethodType
from .constants import DefaultValue, ValidateTrait
from .trait_base import (
from .trait_base import RangeTypes  # noqa: F401, used by TraitsUI
from .trait_errors import TraitError
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_converters import trait_from
from .trait_handler import TraitHandler
from .trait_list_object import TraitListEvent, TraitListObject
from .util.deprecated import deprecated
class TraitCompound(TraitHandler):
    """ Provides a logical-OR combination of other trait handlers.

    This class provides a means of creating complex trait definitions by
    combining several simpler trait definitions. TraitCompound is the
    underlying handler for the general forms of the Trait() function.

    A value is a valid value for a trait attribute based on a TraitCompound
    instance if the value is valid for at least one of the TraitHandler or
    trait objects supplied to the constructor. In addition, if at least one of
    the TraitHandler or trait objects is mapped (e.g., based on a TraitMap or
    TraitPrefixMap instance), then the TraitCompound is also mapped. In this
    case, any non-mapped traits or trait handlers use identity mapping.

    Parameters
    ----------
    *handlers
        Either all TraitHandlers or trait objects to be combined, or a single
        list or tuple of TraitHandlers or trait objects.

    Attributes
    ----------
    handlers : list or tuple
        A list or tuple of TraitHandler or trait objects to be combined.
    """

    def __init__(self, *handlers):
        if len(handlers) == 1 and type(handlers[0]) in SequenceTypes:
            handlers = handlers[0]
        self.handlers = handlers
        self.set_validate()

    def set_validate(self):
        self.is_mapped = False
        self.has_items = False
        self.reversable = True
        post_setattrs = []
        mapped_handlers = []
        validates = []
        fast_validates = []
        slow_validates = []
        for handler in self.handlers:
            fv = getattr(handler, 'fast_validate', None)
            if fv is not None:
                validates.append(handler.validate)
                if fv[0] == ValidateTrait.complex:
                    fast_validates.extend(fv[1])
                else:
                    fast_validates.append(fv)
            else:
                slow_validates.append(handler.validate)
            post_setattr = getattr(handler, 'post_setattr', None)
            if post_setattr is not None:
                post_setattrs.append(post_setattr)
            if handler.is_mapped:
                self.is_mapped = True
                mapped_handlers.append(handler)
            else:
                self.reversable = False
            if handler.has_items:
                self.has_items = True
        self.validates = validates
        self.slow_validates = slow_validates
        if self.is_mapped:
            self.mapped_handlers = mapped_handlers
        elif hasattr(self, 'mapped_handlers'):
            del self.mapped_handlers
        if len(fast_validates) > 0:
            if len(slow_validates) > 0:
                fast_validates.append((ValidateTrait.slow, self))
            self.fast_validate = (ValidateTrait.complex, tuple(fast_validates))
        elif hasattr(self, 'fast_validate'):
            del self.fast_validate
        if len(post_setattrs) > 0:
            self.post_setattrs = post_setattrs
            self.post_setattr = self._post_setattr
        elif hasattr(self, 'post_setattr'):
            del self.post_setattr

    def validate(self, object, name, value):
        for validate in self.validates:
            try:
                return validate(object, name, value)
            except TraitError:
                pass
        return self.slow_validate(object, name, value)

    def slow_validate(self, object, name, value):
        for validate in self.slow_validates:
            try:
                return validate(object, name, value)
            except TraitError:
                pass
        self.error(object, name, value)

    def full_info(self, object, name, value):
        return ' or '.join([x.full_info(object, name, value) for x in self.handlers])

    def info(self):
        return ' or '.join([x.info() for x in self.handlers])

    def mapped_value(self, value):
        for handler in self.mapped_handlers:
            try:
                return handler.mapped_value(value)
            except:
                pass
        return value

    def _post_setattr(self, object, name, value):
        for post_setattr in self.post_setattrs:
            try:
                post_setattr(object, name, value)
                return
            except TraitError:
                pass
        setattr(object, name + '_', value)

    def get_editor(self, trait):
        from traitsui.api import TextEditor, CompoundEditor
        the_editors = [x.get_editor(trait) for x in self.handlers]
        text_editor = TextEditor()
        count = 0
        editors = []
        for editor in the_editors:
            if isinstance(text_editor, editor.__class__):
                count += 1
                if count > 1:
                    continue
            editors.append(editor)
        return CompoundEditor(editors=editors)

    def items_event(self):
        return items_event()