from .base import Renderer
class FullFakeRenderer(FakeRenderer):
    """
    Renderer with the full complement of methods.

    When the following are left undefined, they will be implemented via
    other methods in the class.  They can be defined explicitly for
    more efficient or specialized use within the renderer implementation.
    """

    def draw_line(self, data, coordinates, style, label, mplobj=None):
        self.output += '    draw line with {0} points\n'.format(data.shape[0])

    def draw_markers(self, data, coordinates, style, label, mplobj=None):
        self.output += '    draw {0} markers\n'.format(data.shape[0])

    def draw_path_collection(self, paths, path_coordinates, path_transforms, offsets, offset_coordinates, offset_order, styles, mplobj=None):
        self.output += '    draw path collection with {0} offsets\n'.format(offsets.shape[0])