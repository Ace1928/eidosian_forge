from abc import ABC, abstractmethod
from typing import Any
from qiskit.visualization.pulse_v2 import core
class BasePlotter(ABC):
    """Base class of Qiskit plotter."""

    def __init__(self, canvas: core.DrawerCanvas):
        """Create new plotter.

        Args:
            canvas: Configured drawer canvas object.
        """
        self.canvas = canvas

    @abstractmethod
    def initialize_canvas(self):
        """Format appearance of the canvas."""
        raise NotImplementedError

    @abstractmethod
    def draw(self):
        """Output drawing objects stored in canvas object."""
        raise NotImplementedError

    @abstractmethod
    def get_image(self, interactive: bool=False) -> Any:
        """Get image data to return.

        Args:
            interactive: When set `True` show the circuit in a new window.
                This depends on the matplotlib backend being used supporting this.

        Returns:
            Image data. This depends on the plotter API.
        """
        raise NotImplementedError