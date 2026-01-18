from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter, ScatterPlane
from kivy.properties import ObjectProperty
class ScatterPlaneLayout(ScatterPlane):
    """ScatterPlaneLayout class, see module documentation for more information.

    Similar to ScatterLayout, but based on ScatterPlane - so the input is not
    bounded.

    .. versionadded:: 1.9.0
    """

    def collide_point(self, x, y):
        return True