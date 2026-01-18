from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
class _ChromosomeComponent(Widget):
    """Base class specifying the interface for a component of the system.

    This class should not be instantiated directly, but should be used
    from derived classes.
    """

    def __init__(self):
        """Initialize a chromosome component.

        Attributes:
        - _sub_components -- Any components which are contained under
        this parent component. This attribute should be accessed through
        the add() and remove() functions.

        """
        self._sub_components = []

    def add(self, component):
        """Add a sub_component to the list of components under this item."""
        if not isinstance(component, _ChromosomeComponent):
            raise TypeError(f'Expected a _ChromosomeComponent object, got {component}')
        self._sub_components.append(component)

    def remove(self, component):
        """Remove the specified component from the subcomponents.

        Raises a ValueError if the component is not registered as a
        sub_component.
        """
        try:
            self._sub_components.remove(component)
        except ValueError:
            raise ValueError(f'Component {component} not found in sub_components.') from None

    def draw(self):
        """Draw the specified component."""
        raise AssertionError('Subclasses must implement.')