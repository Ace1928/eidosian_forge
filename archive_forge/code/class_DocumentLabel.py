from os.path import dirname as _dirname
from os.path import splitext as _splitext
import pyglet
from pyglet.text import layout, document, caret
class DocumentLabel(layout.TextLayout):
    """Base label class.

    A label is a layout that exposes convenience methods for manipulating the
    associated document.
    """

    def __init__(self, document=None, x=0, y=0, z=0, width=None, height=None, anchor_x='left', anchor_y='baseline', rotation=0, multiline=False, dpi=None, batch=None, group=None, program=None, init_document=True):
        """Create a label for a given document.

        :Parameters:
            `document` : `AbstractDocument`
                Document to attach to the layout.
            `x` : int
                X coordinate of the label.
            `y` : int
                Y coordinate of the label.
            `z` : int
                Z coordinate of the label.
            `width` : int
                Width of the label in pixels, or None
            `height` : int
                Height of the label in pixels, or None
            `anchor_x` : str
                Anchor point of the X coordinate: one of ``"left"``,
                ``"center"`` or ``"right"``.
            `anchor_y` : str
                Anchor point of the Y coordinate: one of ``"bottom"``,
                ``"baseline"``, ``"center"`` or ``"top"``.
            `rotation`: float
                The amount to rotate the label in degrees. A positive amount
                will be a clockwise rotation, negative values will result in
                counter-clockwise rotation.
            `multiline` : bool
                If True, the label will be word-wrapped and accept newline
                characters.  You must also set the width of the label.
            `dpi` : float
                Resolution of the fonts in this layout.  Defaults to 96.
            `batch` : `~pyglet.graphics.Batch`
                Optional graphics batch to add the label to.
            `group` : `~pyglet.graphics.Group`
                Optional graphics group to use.
            `program` : `~pyglet.graphics.shader.ShaderProgram`
                Optional graphics shader to use. Will affect all glyphs.
            `init_document` : bool
                If True the document will be initialized. If subclassing then
                you may want to avoid duplicate initializations by changing
                to False.
        """
        super().__init__(document, width, height, x, y, z, anchor_x, anchor_y, rotation, multiline, dpi, batch, group, program, init_document=init_document)

    @property
    def text(self):
        """The text of the label.

        :type: str
        """
        return self.document.text

    @text.setter
    def text(self, text):
        self.document.text = text

    @property
    def color(self):
        """Text color.

        Color is a 4-tuple of RGBA components, each in range [0, 255].

        :type: (int, int, int, int)
        """
        return self.document.get_style('color')

    @color.setter
    def color(self, color):
        r, g, b, *a = color
        color = (r, g, b, a[0] if a else 255)
        self.document.set_style(0, len(self.document.text), {'color': color})

    @property
    def opacity(self):
        """Blend opacity.

        This property sets the alpha component of the colour of the label's
        vertices.  With the default blend mode, this allows the layout to be
        drawn with fractional opacity, blending with the background.

        An opacity of 255 (the default) has no effect.  An opacity of 128 will
        make the label appear semi-translucent.

        :type: int
        """
        return self.color[3]

    @opacity.setter
    def opacity(self, alpha):
        if alpha != self.color[3]:
            self.color = list(map(int, (*self.color[:3], alpha)))

    @property
    def font_name(self):
        """Font family name.

        The font name, as passed to :py:func:`pyglet.font.load`.  A list of names can
        optionally be given: the first matching font will be used.

        :type: str or list
        """
        return self.document.get_style('font_name')

    @font_name.setter
    def font_name(self, font_name):
        self.document.set_style(0, len(self.document.text), {'font_name': font_name})

    @property
    def font_size(self):
        """Font size, in points.

        :type: float
        """
        return self.document.get_style('font_size')

    @font_size.setter
    def font_size(self, font_size):
        self.document.set_style(0, len(self.document.text), {'font_size': font_size})

    @property
    def bold(self):
        """Bold font style.

        :type: bool
        """
        return self.document.get_style('bold')

    @bold.setter
    def bold(self, bold):
        self.document.set_style(0, len(self.document.text), {'bold': bold})

    @property
    def italic(self):
        """Italic font style.

        :type: bool
        """
        return self.document.get_style('italic')

    @italic.setter
    def italic(self, italic):
        self.document.set_style(0, len(self.document.text), {'italic': italic})

    def get_style(self, name):
        """Get a document style value by name.

        If the document has more than one value of the named style,
        `pyglet.text.document.STYLE_INDETERMINATE` is returned.

        :Parameters:
            `name` : str
                Style name to query.  See documentation for
                `pyglet.text.layout` for known style names.

        :rtype: object
        """
        return self.document.get_style_range(name, 0, len(self.document.text))

    def set_style(self, name, value):
        """Set a document style value by name over the whole document.

        :Parameters:
            `name` : str
                Name of the style to set.  See documentation for
                `pyglet.text.layout` for known style names.
            `value` : object
                Value of the style.

        """
        self.document.set_style(0, len(self.document.text), {name: value})

    def __del__(self):
        self.delete()