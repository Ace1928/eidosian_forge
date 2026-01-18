from holoviews.core.overlay import CompositeOverlay
from holoviews.core.options import Store
from holoviews.plotting.util import COLOR_ALIASES

    Heuristics to detect if a bokeh option is about interactivity, like
    'selection_alpha'.

    >>> is_interactive_opt('height')
    False
    >>> is_interactive_opt('annular_muted_alpha')
    True
    