import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
class ImageFormatter(Formatter):
    """
    Create a PNG image from source code. This uses the Python Imaging Library to
    generate a pixmap from the source code.

    .. versionadded:: 0.10

    Additional options accepted:

    `image_format`
        An image format to output to that is recognised by PIL, these include:

        * "PNG" (default)
        * "JPEG"
        * "BMP"
        * "GIF"

    `line_pad`
        The extra spacing (in pixels) between each line of text.

        Default: 2

    `font_name`
        The font name to be used as the base font from which others, such as
        bold and italic fonts will be generated.  This really should be a
        monospace font to look sane.

        Default: "Bitstream Vera Sans Mono" on Windows, Courier New on \\*nix

    `font_size`
        The font size in points to be used.

        Default: 14

    `image_pad`
        The padding, in pixels to be used at each edge of the resulting image.

        Default: 10

    `line_numbers`
        Whether line numbers should be shown: True/False

        Default: True

    `line_number_start`
        The line number of the first line.

        Default: 1

    `line_number_step`
        The step used when printing line numbers.

        Default: 1

    `line_number_bg`
        The background colour (in "#123456" format) of the line number bar, or
        None to use the style background color.

        Default: "#eed"

    `line_number_fg`
        The text color of the line numbers (in "#123456"-like format).

        Default: "#886"

    `line_number_chars`
        The number of columns of line numbers allowable in the line number
        margin.

        Default: 2

    `line_number_bold`
        Whether line numbers will be bold: True/False

        Default: False

    `line_number_italic`
        Whether line numbers will be italicized: True/False

        Default: False

    `line_number_separator`
        Whether a line will be drawn between the line number area and the
        source code area: True/False

        Default: True

    `line_number_pad`
        The horizontal padding (in pixels) between the line number margin, and
        the source code area.

        Default: 6

    `hl_lines`
        Specify a list of lines to be highlighted.

        .. versionadded:: 1.2

        Default: empty list

    `hl_color`
        Specify the color for highlighting lines.

        .. versionadded:: 1.2

        Default: highlight color of the selected style
    """
    name = 'img'
    aliases = ['img', 'IMG', 'png']
    filenames = ['*.png']
    unicodeoutput = False
    default_image_format = 'png'

    def __init__(self, **options):
        """
        See the class docstring for explanation of options.
        """
        if not pil_available:
            raise PilNotAvailable('Python Imaging Library is required for this formatter')
        Formatter.__init__(self, **options)
        self.encoding = 'latin1'
        self.styles = dict(self.style)
        if self.style.background_color is None:
            self.background_color = '#fff'
        else:
            self.background_color = self.style.background_color
        self.image_format = get_choice_opt(options, 'image_format', ['png', 'jpeg', 'gif', 'bmp'], self.default_image_format, normcase=True)
        self.image_pad = get_int_opt(options, 'image_pad', 10)
        self.line_pad = get_int_opt(options, 'line_pad', 2)
        fontsize = get_int_opt(options, 'font_size', 14)
        self.fonts = FontManager(options.get('font_name', ''), fontsize)
        self.fontw, self.fonth = self.fonts.get_char_size()
        self.line_number_fg = options.get('line_number_fg', '#886')
        self.line_number_bg = options.get('line_number_bg', '#eed')
        self.line_number_chars = get_int_opt(options, 'line_number_chars', 2)
        self.line_number_bold = get_bool_opt(options, 'line_number_bold', False)
        self.line_number_italic = get_bool_opt(options, 'line_number_italic', False)
        self.line_number_pad = get_int_opt(options, 'line_number_pad', 6)
        self.line_numbers = get_bool_opt(options, 'line_numbers', True)
        self.line_number_separator = get_bool_opt(options, 'line_number_separator', True)
        self.line_number_step = get_int_opt(options, 'line_number_step', 1)
        self.line_number_start = get_int_opt(options, 'line_number_start', 1)
        if self.line_numbers:
            self.line_number_width = self.fontw * self.line_number_chars + self.line_number_pad * 2
        else:
            self.line_number_width = 0
        self.hl_lines = []
        hl_lines_str = get_list_opt(options, 'hl_lines', [])
        for line in hl_lines_str:
            try:
                self.hl_lines.append(int(line))
            except ValueError:
                pass
        self.hl_color = options.get('hl_color', self.style.highlight_color) or '#f90'
        self.drawables = []

    def get_style_defs(self, arg=''):
        raise NotImplementedError('The -S option is meaningless for the image formatter. Use -O style=<stylename> instead.')

    def _get_line_height(self):
        """
        Get the height of a line.
        """
        return self.fonth + self.line_pad

    def _get_line_y(self, lineno):
        """
        Get the Y coordinate of a line number.
        """
        return lineno * self._get_line_height() + self.image_pad

    def _get_char_width(self):
        """
        Get the width of a character.
        """
        return self.fontw

    def _get_char_x(self, charno):
        """
        Get the X coordinate of a character position.
        """
        return charno * self.fontw + self.image_pad + self.line_number_width

    def _get_text_pos(self, charno, lineno):
        """
        Get the actual position for a character and line position.
        """
        return (self._get_char_x(charno), self._get_line_y(lineno))

    def _get_linenumber_pos(self, lineno):
        """
        Get the actual position for the start of a line number.
        """
        return (self.image_pad, self._get_line_y(lineno))

    def _get_text_color(self, style):
        """
        Get the correct color for the token from the style.
        """
        if style['color'] is not None:
            fill = '#' + style['color']
        else:
            fill = '#000'
        return fill

    def _get_style_font(self, style):
        """
        Get the correct font for the style.
        """
        return self.fonts.get_font(style['bold'], style['italic'])

    def _get_image_size(self, maxcharno, maxlineno):
        """
        Get the required image size.
        """
        return (self._get_char_x(maxcharno) + self.image_pad, self._get_line_y(maxlineno + 0) + self.image_pad)

    def _draw_linenumber(self, posno, lineno):
        """
        Remember a line number drawable to paint later.
        """
        self._draw_text(self._get_linenumber_pos(posno), str(lineno).rjust(self.line_number_chars), font=self.fonts.get_font(self.line_number_bold, self.line_number_italic), fill=self.line_number_fg)

    def _draw_text(self, pos, text, font, **kw):
        """
        Remember a single drawable tuple to paint later.
        """
        self.drawables.append((pos, text, font, kw))

    def _create_drawables(self, tokensource):
        """
        Create drawables for the token content.
        """
        lineno = charno = maxcharno = 0
        for ttype, value in tokensource:
            while ttype not in self.styles:
                ttype = ttype.parent
            style = self.styles[ttype]
            value = value.expandtabs(4)
            lines = value.splitlines(True)
            for i, line in enumerate(lines):
                temp = line.rstrip('\n')
                if temp:
                    self._draw_text(self._get_text_pos(charno, lineno), temp, font=self._get_style_font(style), fill=self._get_text_color(style))
                    charno += len(temp)
                    maxcharno = max(maxcharno, charno)
                if line.endswith('\n'):
                    charno = 0
                    lineno += 1
        self.maxcharno = maxcharno
        self.maxlineno = lineno

    def _draw_line_numbers(self):
        """
        Create drawables for the line numbers.
        """
        if not self.line_numbers:
            return
        for p in xrange(self.maxlineno):
            n = p + self.line_number_start
            if n % self.line_number_step == 0:
                self._draw_linenumber(p, n)

    def _paint_line_number_bg(self, im):
        """
        Paint the line number background on the image.
        """
        if not self.line_numbers:
            return
        if self.line_number_fg is None:
            return
        draw = ImageDraw.Draw(im)
        recth = im.size[-1]
        rectw = self.image_pad + self.line_number_width - self.line_number_pad
        draw.rectangle([(0, 0), (rectw, recth)], fill=self.line_number_bg)
        draw.line([(rectw, 0), (rectw, recth)], fill=self.line_number_fg)
        del draw

    def format(self, tokensource, outfile):
        """
        Format ``tokensource``, an iterable of ``(tokentype, tokenstring)``
        tuples and write it into ``outfile``.

        This implementation calculates where it should draw each token on the
        pixmap, then calculates the required pixmap size and draws the items.
        """
        self._create_drawables(tokensource)
        self._draw_line_numbers()
        im = Image.new('RGB', self._get_image_size(self.maxcharno, self.maxlineno), self.background_color)
        self._paint_line_number_bg(im)
        draw = ImageDraw.Draw(im)
        if self.hl_lines:
            x = self.image_pad + self.line_number_width - self.line_number_pad + 1
            recth = self._get_line_height()
            rectw = im.size[0] - x
            for linenumber in self.hl_lines:
                y = self._get_line_y(linenumber - 1)
                draw.rectangle([(x, y), (x + rectw, y + recth)], fill=self.hl_color)
        for pos, value, font, kw in self.drawables:
            draw.text(pos, value, font=font, **kw)
        im.save(outfile, self.image_format.upper())