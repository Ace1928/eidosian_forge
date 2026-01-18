from collections.abc import Iterable, Mapping
from itertools import chain
from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import InstanceDict, TypedTuple
from .widget import register, widget_serialization
from .widget_int import SliderStyle
from .docutils import doc_subst
from traitlets import (Unicode, Bool, Int, Any, Dict, TraitError, CaselessStrEnum,
class _Selection(DescriptionWidget, ValueWidget, CoreWidget):
    """Base class for Selection widgets

    ``options`` can be specified as a list of values or a list of (label, value)
    tuples. The labels are the strings that will be displayed in the UI,
    representing the actual Python choices, and should be unique.
    If labels are not specified, they are generated from the values.

    When programmatically setting the value, a reverse lookup is performed
    among the options to check that the value is valid. The reverse lookup uses
    the equality operator by default, but another predicate may be provided via
    the ``equals`` keyword argument. For example, when dealing with numpy arrays,
    one may set equals=np.array_equal.
    """
    value = Any(None, help='Selected value', allow_none=True)
    label = Unicode(None, help='Selected label', allow_none=True)
    index = Int(None, help='Selected index', allow_none=True).tag(sync=True)
    options = Any((), help='Iterable of values, (label, value) pairs, or Mapping between labels and values that the user can select.\n\n    The labels are the strings that will be displayed in the UI, representing the\n    actual Python choices, and should be unique.\n    ')
    _options_full = None
    _options_labels = TypedTuple(trait=Unicode(), read_only=True, help='The labels for the options.').tag(sync=True)
    disabled = Bool(help='Enable or disable user changes').tag(sync=True)

    def __init__(self, *args, **kwargs):
        self.equals = kwargs.pop('equals', lambda x, y: x == y)
        self._initializing_traits_ = True
        kwargs['options'] = _exhaust_iterable(kwargs.get('options', ()))
        self._options_full = _make_options(kwargs['options'])
        self._propagate_options(None)
        if 'index' not in kwargs and 'value' not in kwargs and ('label' not in kwargs):
            options = self._options_full
            nonempty = len(options) > 0
            kwargs['index'] = 0 if nonempty else None
            kwargs['label'], kwargs['value'] = options[0] if nonempty else (None, None)
        super().__init__(*args, **kwargs)
        self._initializing_traits_ = False

    @validate('options')
    def _validate_options(self, proposal):
        proposal.value = _exhaust_iterable(proposal.value)
        self._options_full = _make_options(proposal.value)
        return proposal.value

    @observe('options')
    def _propagate_options(self, change):
        """Set the values and labels, and select the first option if we aren't initializing"""
        options = self._options_full
        self.set_trait('_options_labels', tuple((i[0] for i in options)))
        self._options_values = tuple((i[1] for i in options))
        if self.index is None:
            return
        if self._initializing_traits_ is not True:
            if len(options) > 0:
                if self.index == 0:
                    self._notify_trait('index', 0, 0)
                else:
                    self.index = 0
            else:
                self.index = None

    @validate('index')
    def _validate_index(self, proposal):
        if proposal.value is None or 0 <= proposal.value < len(self._options_labels):
            return proposal.value
        else:
            raise TraitError('Invalid selection: index out of bounds')

    @observe('index')
    def _propagate_index(self, change):
        """Propagate changes in index to the value and label properties"""
        label = self._options_labels[change.new] if change.new is not None else None
        value = self._options_values[change.new] if change.new is not None else None
        if self.label is not label:
            self.label = label
        if self.value is not value:
            self.value = value

    @validate('value')
    def _validate_value(self, proposal):
        value = proposal.value
        try:
            return findvalue(self._options_values, value, self.equals) if value is not None else None
        except ValueError:
            raise TraitError('Invalid selection: value not found')

    @observe('value')
    def _propagate_value(self, change):
        if change.new is None:
            index = None
        elif self.index is not None and self.equals(self._options_values[self.index], change.new):
            index = self.index
        else:
            index = self._options_values.index(change.new)
        if self.index != index:
            self.index = index

    @validate('label')
    def _validate_label(self, proposal):
        if proposal.value is not None and proposal.value not in self._options_labels:
            raise TraitError('Invalid selection: label not found')
        return proposal.value

    @observe('label')
    def _propagate_label(self, change):
        if change.new is None:
            index = None
        elif self.index is not None and self._options_labels[self.index] == change.new:
            index = self.index
        else:
            index = self._options_labels.index(change.new)
        if self.index != index:
            self.index = index

    def _repr_keys(self):
        keys = super()._repr_keys()
        for key in sorted(chain(keys, ('options',))):
            if key == 'index' and self.index == 0:
                continue
            yield key