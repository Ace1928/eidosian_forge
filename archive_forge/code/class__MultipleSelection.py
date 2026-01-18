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
class _MultipleSelection(DescriptionWidget, ValueWidget, CoreWidget):
    """Base class for multiple Selection widgets

    ``options`` can be specified as a list of values, list of (label, value)
    tuples, or a dict of {label: value}. The labels are the strings that will be
    displayed in the UI, representing the actual Python choices, and should be
    unique. If labels are not specified, they are generated from the values.

    When programmatically setting the value, a reverse lookup is performed
    among the options to check that the value is valid. The reverse lookup uses
    the equality operator by default, but another predicate may be provided via
    the ``equals`` keyword argument. For example, when dealing with numpy arrays,
    one may set equals=np.array_equal.
    """
    value = TypedTuple(trait=Any(), help='Selected values')
    label = TypedTuple(trait=Unicode(), help='Selected labels')
    index = TypedTuple(trait=Int(), help='Selected indices').tag(sync=True)
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
        super().__init__(*args, **kwargs)
        self._initializing_traits_ = False

    @validate('options')
    def _validate_options(self, proposal):
        proposal.value = _exhaust_iterable(proposal.value)
        self._options_full = _make_options(proposal.value)
        return proposal.value

    @observe('options')
    def _propagate_options(self, change):
        """Unselect any option"""
        options = self._options_full
        self.set_trait('_options_labels', tuple((i[0] for i in options)))
        self._options_values = tuple((i[1] for i in options))
        if self._initializing_traits_ is not True:
            self.index = ()

    @validate('index')
    def _validate_index(self, proposal):
        """Check the range of each proposed index."""
        if all((0 <= i < len(self._options_labels) for i in proposal.value)):
            return proposal.value
        else:
            raise TraitError('Invalid selection: index out of bounds')

    @observe('index')
    def _propagate_index(self, change):
        """Propagate changes in index to the value and label properties"""
        label = tuple((self._options_labels[i] for i in change.new))
        value = tuple((self._options_values[i] for i in change.new))
        if self.label != label:
            self.label = label
        if self.value != value:
            self.value = value

    @validate('value')
    def _validate_value(self, proposal):
        """Replace all values with the actual objects in the options list"""
        try:
            return tuple((findvalue(self._options_values, i, self.equals) for i in proposal.value))
        except ValueError:
            raise TraitError('Invalid selection: value not found')

    @observe('value')
    def _propagate_value(self, change):
        index = tuple((self._options_values.index(i) for i in change.new))
        if self.index != index:
            self.index = index

    @validate('label')
    def _validate_label(self, proposal):
        if any((i not in self._options_labels for i in proposal.value)):
            raise TraitError('Invalid selection: label not found')
        return proposal.value

    @observe('label')
    def _propagate_label(self, change):
        index = tuple((self._options_labels.index(i) for i in change.new))
        if self.index != index:
            self.index = index

    def _repr_keys(self):
        keys = super()._repr_keys()
        yield from sorted(chain(keys, ('options',)))